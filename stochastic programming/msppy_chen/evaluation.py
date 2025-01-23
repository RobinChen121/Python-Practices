"""
created on 2025/1/19, 23:36
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@desp:

"""
import numpy
import multiprocessing
import pandas


class _Evaluation(object):
    """Evaluation base class.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.

    Attributes
    ----------
    db: list
        The deterministic bounds.

    pv: list
        The simulated policy values.

    epv: float
        The exact value of expected policy value (only available for
        approximation model).

    CI: tuple
        The CI of simulated policy values.

    gap: float
        The gap between upper end of the CI and deterministic bound.

    stage_cost: dataframe
        The cost of individual stage models.

    solution: dataframe
        The solution of queried variables.

    n_sample_paths: int
        The number of sample paths to evaluate policy.

    sample_paths_idx: list
        The index list of exhaustive sample paths if simulation is turned off.

    markovian_samples:
        The simulated Markovian type samples.

    markovian_idx: list
        The Markov state that is the closest to the markovian_samples.
    """

    def __init__(self, MSP):
        self.MSP = MSP
        self.db = MSP.db
        self.pv = None
        self.CI = None
        self.epv = None
        self.gap = None
        self.stage_cost = None
        self.solution = None
        self.n_sample_paths = None
        self.sample_path_idx = None
        self.markovian_idx = None
        self.markovian_samples = None
        self.solve_true = False

    def _compute_gap(self):
        if self.MSP.measure != 'risk neutral':
            self.gap = -1
            return
        try:
            MSP = self.MSP
            if self.CI is not None:
                if MSP.sense == 1:
                    self.gap = abs((self.CI[1] - self.db) / self.db)
                else:
                    self.gap = abs((self.db - self.CI[0]) / self.db)
            elif self.epv is not None:
                self.gap = abs((self.epv - self.db) / self.db)
            else:
                self.gap = abs((self.pv[0] - self.db) / self.db)
        except ZeroDivisionError:
            self.gap = -1

    def _compute_sample_path_idx_and_markovian_path(self):
        pass

    def run(
            self,
            n_simulations,
            percentile=95,
            query=None,
            query_T=None,
            query_dual=None,
            query_stage_cost=False,
            n_processes=1, ):
        """Run a Monte Carlo simulation to evaluate the policy.

        Parameters
        ----------
        n_simulations: int/-1
            If int: the number of simulations;
            If -1: exhuastive evaluation.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        query_dual: list, optional (default=None)
            The names of constraints whose dual variables are intended to query.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        n_processes: int, optional (default=1)
            The number of processes to run the simulation.

        T: int, optional (default=None)
            For infinite horizon problem, the number stages to evaluate the policy.
        """
        MSP = self.MSP
        query_T = query_T if query_T else MSP.T
        if not MSP._flag_infinity:
            from msppy.solver import SDDP
            self.solver = SDDP(MSP)
        else:
            from msppy.solver import PSDDP
            self.solver = PSDDP(MSP)
            self.solver.forward_T = query_T
        self.n_simulations = n_simulations
        self._compute_sample_path_idx_and_markovian_path(query_T)
        self.pv = numpy.zeros(self.n_sample_paths)
        stage_cost = solution = solution_dual = None
        if query_stage_cost:
            stage_cost = [
                multiprocessing.RawArray("d", [0] * (query_T))
                for _ in range(self.n_sample_paths)
            ]
        if query is not None:
            solution = {
                item: [
                    multiprocessing.RawArray("d", [0] * (query_T))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query
            }
        if query_dual is not None:
            solution_dual = {
                item: [
                    multiprocessing.RawArray("d", [0] * (query_T))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query_dual
            }
        n_processes = min(self.n_sample_paths, n_processes)
        jobs = allocate_jobs(self.n_sample_paths, n_processes)
        pv = multiprocessing.Array("d", [0] * self.n_sample_paths)
        procs = [None] * n_processes
        for p in range(n_processes):
            procs[p] = multiprocessing.Process(
                target=self.run_single,
                args=(pv, jobs[p], query, query_dual, query_stage_cost, stage_cost,
                      solution, solution_dual)
            )
            procs[p].start()
        for proc in procs:
            proc.join()
        if self.n_simulations != 1:
            self.pv = [item for item in pv]
        else:
            self.pv = pv[0]
        if self.n_simulations == -1:
            self.epv = numpy.dot(
                pv,
                [
                    MSP._compute_weight_sample_path(self.sample_path_idx[j])
                    for j in range(self.n_sample_paths)
                ],
            )
        if self.n_simulations not in [-1, 1]:
            self.CI = compute_CI(self.pv, percentile)
        self._compute_gap()
        if query is not None:
            self.solution = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution.items()
            }
        if query_dual is not None:
            self.solution_dual = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution_dual.items()
            }
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(numpy.array(stage_cost))

    def run_single(self, pv, jobs, query=None, query_dual=None,
                   query_stage_cost=False, stage_cost=None,
                   solution=None, solution_dual=None):
        random_state = numpy.random.RandomState([2 ** 32 - 1, jobs[0]])
        MSP = self.MSP
        markovian_samples = markovian_idices = None
        solver = self.solver
        if MSP._type == "Markovian" and self.solve_true:
            markovian_samples = MSP.Markovian_uncertainty(
                random_state, len(jobs))
            markovian_idices = numpy.zeros([len(jobs), solver.forward_T], dtype=int)
            for t in range(1, solver.forward_T):
                idx, _ = solver._compute_idx(t)
                dist = numpy.empty([len(jobs), MSP.n_Markov_states[idx]])
                for i, markov_state in enumerate(MSP.Markov_states[idx]):
                    temp = markovian_samples[:, t, :] - markov_state
                    dist[:, i] = numpy.sum(temp ** 2, axis=1)
                markovian_idices[:, t] = numpy.argmin(dist, axis=1)

        for idx, j in enumerate(jobs):
            sample_path_idx = (self.sample_path_idx[j]
                               if self.sample_path_idx is not None else None)
            markovian_idx = (markovian_idices[idx]
                             if markovian_idices is not None else None)
            markovian_sample = (markovian_samples[idx]
                                if markovian_samples is not None else None)
            result = self.solver._forward(
                random_state=random_state,
                sample_path_idx=sample_path_idx,
                markovian_idx=markovian_idx,
                markovian_samples=markovian_sample,
                solve_true=self.solve_true,
                query=query,
                query_dual=query_dual,
                query_stage_cost=query_stage_cost
            )
            if query is not None:
                for item in query:
                    for i in range(len(solution[item][0])):
                        solution[item][j][i] = result['solution'][item][i]
            if query_dual is not None:
                for item in solution_dual:
                    for i in range(len(solution_dual[item][0])):
                        solution_dual[item][j][i] = result['solution_dual'][item][i]
            if query_stage_cost:
                for i in range(len(stage_cost[0])):
                    stage_cost[j][i] = result['stage_cost'][i]
            pv[j] = result['pv']


class Evaluation(_Evaluation):
    __doc__ = _Evaluation.__doc__

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)

    def _compute_sample_path_idx_and_markovian_path(self, T):
        if self.n_simulations == -1:
            self.n_sample_paths, self.sample_path_idx = self.MSP._enumerate_sample_paths(T - 1)
        else:
            self.n_sample_paths = self.n_simulations


class EvaluationTrue(Evaluation):
    __doc__ = Evaluation.__doc__

    def run(self, *args, **kwargs):
        MSP = self.MSP
        if MSP.__class__.__name__ == 'MSIP':
            MSP._back_binarize()
        # discrete finite model should call evaluate instead
        if (
                MSP._type in ["stage-wise independent", "Markov chain"]
                and MSP._individual_type == "original"
                and not hasattr(MSP, "bin_stage")
        ):
            return super().run(*args, **kwargs)
        return _Evaluation.run(self, *args, **kwargs)

    def _compute_sample_path_idx_and_markovian_path(self, T):
        MSP = self.MSP
        if (
                MSP._type in ["stage-wise independent", "Markov chain"]
                and MSP._individual_type == "original"
                and not hasattr(MSP, "bin_stage")
        ):
            return super()._compute_sample_path_idx_and_markovian_path(T)
        self.n_sample_paths = self.n_simulations
        self.solve_true = True
