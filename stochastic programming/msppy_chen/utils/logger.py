"""
Created on 2025/1/17, 23:18 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""
import logging


class Logger:
    """
    Basic log class

    Args:
        logToFile_flag: bool, whether to log in a file
        logToConsole_flag: bool, whether to log to console
        directory: directory address for the log file

    """

    def __init__(self, logToFile_flag: bool = False, logToConsole_flag: bool = False, directory: str = ''):
        name = ""
        logger = logging.getLogger(
            name)  # getLogger() return a logger with the specified name or, if name is None, return the root logger of the hierarchy.
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        if logToFile_flag != 0:
            handler = logging.FileHandler(directory + name + ".log", mode="w")
            logger.addHandler(handler)
        if logToConsole_flag != 0:
            streamHandler = logging.StreamHandler()
            logger.addHandler(streamHandler)
        self.logger = logger
        self.time = 0
        self.n_slots = 50

    def footer(self, *args) -> None:
        self.logger.info("-" * self.n_slots)
        self.logger.info("Time: {} seconds".format(self.time))

class LoggerSDDP(Logger):
    """
    log class for SDDP

    Args:
        percentile: the percentile used to compute confidence interval
        n_process: the number of cpu processors
        **kwargs: logToFile_flag: bool, whether to log in a file
                  logToConsole_flag: bool, whether to log to console
                  directory: directory address for the log file
    Attributes:
        n_slots: width of the output string
    """

    def __init__(self, percentile: float, n_processes: int, **kwargs):
        self.percentile = percentile
        self.n_processes = n_processes
        super().__init__(**kwargs)
        self.n_slots = 100 if self.n_processes > 1 else 64

    def __repr__(self):
        return "SDDP"

    def header(self) -> None:
        """
            header of the SDDP logger
        """
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}}".format("SDDP Solver, Zhen Chen", width = self.n_slots)
        )
        self.logger.info("-" * self.n_slots)
        if self.n_processes > 1:
            self.logger.info(
                "{:<10s}{:^30s}{:^50s}{:>10s}" # s is not necessary
                .format(
                    "Iteration",
                    "Bound",
                    "Value {}%CI ({}processor)".format(self.percentile, self.n_processes),
                    "Time"
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value",
                    "Time"
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration, obj_bound: float, time, policy_value: float = None, CI: tuple = None):
        """
            body text of the logger
        Args:
            iteration: iteration index
            obj_bound: objective bound
            time: elapsed time
            policy_value: policy value at the current iteration
            CI: confidence interval
        """
        if self.n_processes > 1:
            self.logger.info(
                "{:>12d}{:>20f}{:>19f}, {:<19f}{:>12f}".format(
                    iteration, obj_bound, CI[0], CI[1], time
                )
            )
        else:
            self.logger.info(
                "{:>12d}{:>20f}{:>20f}{:>12f}".format(
                    iteration, obj_bound, policy_value, time
                )
            )
        self.time += time

    def footer(self, reason: str) -> None:
        super().footer()
        self.logger.info("Algorithm stops since " + reason)

class LoggerEvaluation(Logger):
    def __init__(self, percentile: float, n_simulations: int, **kwargs):
        self.percentile = percentile
        self.n_simulations = n_simulations
        self.n_slots = 76 if self.n_simulations in [-1,1] else 96
        super().__init__(**kwargs)

    def __repr__(self):
        return "Evaluation"

    def header(self) -> None:
        """
            header of the evaluation logger
        """
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}s}".format(
                "Evaluation for approximation model, Zhen Chen",
                width = self.n_slots
            )
        )
        self.logger.info("-" * self.n_slots)
        if self.n_simulations not in [-1,1]:
            self.logger.info(
                "{:>12s}{:>20s}{:^50s}{:>12s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value {}% CI({} simulations)".format(self.percentile, self.n_simulations),
                    "Time",
                    "Gap",
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value",
                    "Time",
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration, obj_bound: float, time: float, policy_value: float = None, CI: list = None, gap: float = None):
        """
            body text of the logger
        Args:
            iteration: iteration index
            obj_bound: objective bound
            time: elapsed time
            policy_value: policy value at the current iteration
            CI: confidence interval
            gap:
        """
        if self.n_simulations > 1:
            format_ = "{:>12d}{:>20f}{:>19f}, {:<19f}{:>12f}"
            if gap in [-1, None]:
                format_ += "{:>12}"
            else:
                format_ += "{:>12.2%}"
            self.logger.info(
                format_.format(
                    iteration, obj_bound, CI[0], CI[1], time, gap
                )
            )
        else:
            format_ = "{:>12d}{:>20f}{:>20f}{:>12f}"
            if gap in [-1, None]:
                format_ += "{:>12}"
            else:
                format_ += "{:>12.2%}"
            self.logger.info(
                format_.format(
                    iteration, obj_bound, policy_value, time, gap
                )
            )
        self.time += time

class LoggerComparison(Logger):
    def __init__(self, percentile: float, n_simulations: int, **kwargs):
        self.percentile = percentile
        self.n_simulations = n_simulations
        self.n_slots = 64 if self.n_simulations in [-1,1] else 84
        super().__init__(**kwargs)

    def __repr__(self):
        return "Comparison"

    def header(self) -> None:
        """
            header of the comparison logger
        """
        assert self.n_simulations != 1
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}s}".format(
                "Comparison for approximation model, Zhen Chen",
                width = self.n_slots
            )
        )
        self.logger.info("-" * self.n_slots)
        if self.n_simulations != -1:
            self.logger.info(
                "{:>12s}{:>20s}{:^40s}{:>12s}"
                .format(
                    "Iteration",
                    "Reference iter.",
                    "Difference {}% CI ({})".format(self.percentile,self.n_simulations),
                    "Time",
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Reference iter.",
                    "Difference",
                    "Time",
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration: int, ref_iteration: int, time: float, diff_CI: list = None, diff: float = None):
        assert self.n_simulations != 1
        if self.n_simulations != -1:
            self.logger.info(
                "{:>12d}{:>20d}{:>19f}, {:<19f}{:>12f}".format(
                    iteration, ref_iteration, diff_CI[0], diff_CI[1], time
                )
            )
        else:
            self.logger.info(
                "{:>12d}{:>20d}{:>20f}{:>12f}".format(
                    iteration, ref_iteration, diff, time
                )
            )
        self.time += time

if __name__ == '__main__':
    test = Logger()
