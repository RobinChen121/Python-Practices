"""
Created on 2025/1/10, 21:48

@author: Zhen Chen.

@Python version: 3.10

@disp:  Different classed of stochastic programming solvers.

"""
from msm import MSP


class Extensive:
    """
    Extensive solver class.

    Can solve:

    1. small-scale stage-wise independent finite discrete risk neutral problem;

    2. small-scale Markov chain risk neutral problem.

    Parameters:
    ----------
    msp: A multi-stage stochastic program object.

    Attributes
    ----------
    extensive_model:
        The constructed extensive model

    solving_time:
        The time cost in solving extensive model

    construction_time:
        The time cost in constructing extensive model
    """

    def __init__(self, msp: MSP) -> None:
        self.msp = msp
        self.solving_time = None
        self.construction_time = None
        self.total_time = None
        self.type = msp.type

    def __getattr__(self, name: any) -> any:
        """
        Called when the default attribute access fails with an AttributeError.

        Args:
            name: the attribute

        Returns:
            The attribute of this model.
        """
        try:
            return getattr(self.extensive_model, name)
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

    def solve(self, start=0, flag_rolling=0, **kwargs):
        """Call extensive solver to solve the discretized problem. It will first
        construct the extensive model and then call Gurobi solver to solve it.

        Parameters
        ----------
        **kwargs: optional
            Gurobipy attributes to specify on extensive model.
        """
        # extensive solver is able to solve MSLP with CTG or without CTG
        self.MSP._check_individual_stage_models()
        self.MSP._check_multistage_model()

        construction_start_time = time.time()

        self.extensive_model = gurobipy.Model()
        self.extensive_model.modelsense = self.MSP.sense
        self.start = start

        for k, v in kwargs.items():
            setattr(self.extensive_model.Params, k, v)
        self._construct_extensive(flag_rolling)
        construction_end_time = time.time()
        self.construction_time = construction_end_time - construction_start_time
        solving_start_time = time.time()
        self.extensive_model.optimize()
        solving_end_time = time.time()
        self.solving_time = solving_end_time - solving_start_time
        self.total_time = self.construction_time + self.solving_time
        return self.extensive_model.objVal

    def _get_varname(self):
        if type(self.MSP.models[self.start]) != list:
            names = [var.varname for var in self.MSP.models[self.start].getVars()]
        else:
            names = [var.varname for var in self.MSP.models[self.start][0].getVars()]
        return names

    def _get_first_stage_vars(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            vars = {name:self.extensive_model.getVarByName(name+'(0,)')
                for name in names}
        else:
            vars = {name:self.extensive_model.getVarByName(name+'((0,),(0,))')
                for name in names}
        return vars

    def _get_first_stage_states(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            states = {name:self.extensive_model.getVarByName(name+'(0,)')
                for name in names}
        else:
            states = {name:self.extensive_model.getVarByName(name+'((0,),(0,))')
                for name in names}
        return states

    @property
    def first_stage_solution(self):
        """the obtained solution in the first stage"""
        states = self._get_first_stage_states()
        return {k:v.X for k,v in states.items()}

    @property
    def first_stage_all_solution(self):
        vars = self._get_first_stage_vars()
        return {k:v.X for k,v in vars.items()}

    @property
    def first_stage_cost(self):
        vars = self._get_first_stage_vars()
        return sum(v.obj*v.X for k,v in vars.items())

    def _construct_extensive(self, flag_rolling):
        ## Construct extensive model
        MSP = self.MSP
        T = MSP.T
        start = self.start
        n_Markov_states = MSP.n_Markov_states
        n_samples = (
            [MSP.models[t].n_samples for t in range(T)]
            if n_Markov_states == 1
            else [MSP.models[t][0].n_samples for t in range(T)]
        )
        n_states = MSP.n_states
        # check if CTG variable is added or not
        initial_model = (
            MSP.models[start] if n_Markov_states == 1 else MSP.models[start][0]
        )
        flag_CTG = 1 if initial_model.alpha is not None else -1
        # |       stage 0       |        stage 1       | ... |       stage T-1      |
        # |local_copies, states | local_copies, states | ... | local_copies, states |
        # |local_copies,        | local_copies,        | ... | local_copies, states |
        # extensive formulation only includes necessary variables
        states = None
        sample_paths = None
        if flag_CTG == 1:
            stage_cost = None
        for t in reversed(range(start,T)):
            M = [MSP.models[t]] if n_Markov_states == 1 else MSP.models[t]
            # stage T-1 needs to add the states. sample path corresponds to
            # current node.
            if t == T-1:
                _, sample_paths = MSP._enumerate_sample_paths(t,start,flag_rolling)
                states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]
            # new_states is the local_copies. new_sample_paths corresponds to
            # previous node
            if t != start:
                temp, new_sample_paths = MSP._enumerate_sample_paths(t-1,start,flag_rolling)
                new_states = [
                    self.extensive_model.addVars(new_sample_paths)
                    for _ in range(n_states[t-1])
                ]
                if flag_CTG == 1:
                    new_stage_cost = {
                        new_sample_path: 0
                        for new_sample_path in new_sample_paths
                    }
            else:
                new_states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]

            for j in range(n_samples[t]):
                for k, m in enumerate(M):
                    # copy information from model in scenario j and markov state
                    # k.
                    m._update_uncertainty(j)
                    m.update()
                    # compute sample paths that go through the current node
                    current_sample_paths = (
                        [
                            item
                            for item in sample_paths
                            if item[0][t-start] == j and item[1][t-start] == k
                        ]
                        if n_Markov_states != 1
                        else [item for item in sample_paths if item[t-start] == j]
                    )
                    # when the sample path is too long, change the name of variables

                    controls_ = m.controls
                    states_ = m.states
                    local_copies_ = m.local_copies
                    controls_dict = {v: i for i, v in enumerate(controls_)}
                    states_dict = {v: i for i, v in enumerate(states_)}
                    local_copies_dict = {
                        v: i for i, v in enumerate(local_copies_)
                    }

                    for current_sample_path in current_sample_paths:
                        flag_reduced_name = 0
                        if len(str(current_sample_path)) > 100:
                            flag_reduced_name = 1
                        if t != start:
                            # compute sample paths that go through the
                            # ancester node
                            past_sample_path = (
                                current_sample_path[:-1]
                                if n_Markov_states == 1
                                else (
                                    current_sample_path[0][:-1],
                                    current_sample_path[1][:-1],
                                )
                            )
                        else:
                            past_sample_path = current_sample_path

                        if flag_CTG == -1 or t == start:
                            weight = MSP.discount ** (
                                (t - start)
                            ) * MSP._compute_weight_sample_path(
                                current_sample_path, start
                            )
                        else:
                            currentWeight = MSP._compute_current_weight_sample_path(
                                current_sample_path)

                        for i in range(n_states[t]):
                            obj = (
                                states_[i].obj * numpy.array(weight)
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            states[i][current_sample_path].lb = states_[i].lb
                            states[i][current_sample_path].ub = states_[i].ub
                            states[i][current_sample_path].obj = obj
                            states[i][current_sample_path].vtype = states_[
                                i
                            ].vtype
                            if flag_reduced_name == 0:
                                states[i][current_sample_path].varName = states_[
                                    i
                                ].varName + str(current_sample_path).replace(
                                    " ", ""
                                )
                            # cost-to-go update
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    states[i][current_sample_path]
                                    * states_[i].obj
                                    * currentWeight
                                )

                        if t == start:
                            for i in range(n_states[t]):
                                new_states[i][current_sample_path].lb = local_copies_[i].lb
                                new_states[i][current_sample_path].ub = local_copies_[i].ub
                                new_states[i][current_sample_path].obj = local_copies_[i].obj
                                new_states[i][current_sample_path].vtype = local_copies_[i].vtype
                                if flag_reduced_name == 0:
                                    new_states[i][current_sample_path].varName = local_copies_[i].varname + str(current_sample_path).replace(" ", "")
                        # copy local variables
                        controls = [None for _ in range(len(controls_))]
                        for i, var in enumerate(controls_):
                            obj = (
                                var.obj * weight
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            controls[i] = self.extensive_model.addVar(
                                lb=var.lb,
                                ub=var.ub,
                                obj=obj,
                                vtype=var.vtype,
                                name=(
                                    var.varname
                                    + str(current_sample_path).replace(" ", "")
                                    if flag_reduced_name == 0
                                    else ""
                                ),
                            )
                            # cost-to-go update
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    controls[i] * var.obj * currentWeight
                                )
                        # self.extensive_model.update()
                        # add constraints
                        if t != T - 1 and flag_CTG == 1:
                            self.extensive_model.addConstr(
                                MSP.sense
                                * (
                                    controls[
                                        controls_dict[m.getVarByName("alpha")]
                                    ]
                                    - stage_cost[current_sample_path]
                                )
                                >= 0
                            )
                        for constr_ in m.getConstrs():
                            rhs_ = constr_.rhs
                            expr_ = m.getRow(constr_)
                            lhs = gurobipy.LinExpr()
                            for i in range(expr_.size()):

                                if expr_.getVar(i) in controls_dict.keys():
                                    pos = controls_dict[expr_.getVar(i)]
                                    lhs += expr_.getCoeff(i) * controls[pos]
                                elif expr_.getVar(i) in states_dict.keys():
                                    pos = states_dict[expr_.getVar(i)]
                                    lhs += (
                                        expr_.getCoeff(i)
                                        * states[pos][current_sample_path]
                                    )
                                elif (
                                    expr_.getVar(i) in local_copies_dict.keys()
                                ):
                                    pos = local_copies_dict[expr_.getVar(i)]
                                    if t != start:
                                        lhs += (
                                            expr_.getCoeff(i)
                                            * new_states[pos][past_sample_path]
                                        )
                                    else:
                                        lhs += (
                                            expr_.getCoeff(i)
                                            * new_states[pos][current_sample_path]
                                        )
                            #! end expression loop
                            self.extensive_model.addConstr(
                                lhs=lhs, sense=constr_.sense, rhs=rhs_
                            )
                        #! end copying the constraints
                    #! end MC loop
                #! end scenarios loop
            #! end scenario loop
            states = new_states
            if flag_CTG == 1:
                stage_cost = new_stage_cost
            sample_paths = new_sample_paths