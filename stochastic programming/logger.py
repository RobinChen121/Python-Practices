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
        logFile: bool, whether to log in a file
        logToConsole: bool, whether to log to console
        directory: directory address for the log file

    """

    def __init__(self, logFile: bool = False, logToConsole: bool = False, directory: str = ''):
        name = ""
        logger = logging.getLogger(
            name)  # getLogger() return a logger with the specified name or, if name is None, return the root logger of the hierarchy.
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        if logFile != 0:
            handler = logging.FileHandler(directory + name + ".log", mode="w")
            logger.addHandler(handler)
        if logToConsole != 0:
            streamHandler = logging.StreamHandler()
            logger.addHandler(streamHandler)
        self.logger = logger
        self.time = 0


class LoggerSDDP(Logger):
    """
    log class for SDDP

    Args:
        percentile: the percentile used to compute confidence interval
        n_process: the number of cpu processors
        **kwargs: logFile: bool, whether to log in a file
                  logToConsole: bool, whether to log to console
                  directory: directory address for the log file
    Attributes:
        n_slots: width of the output string
    """

    def __init__(self, percentile: float, n_processes: int, **kwargs):
        self.percentile = percentile
        self.n_processes = n_processes
        super().__init__(**kwargs)
        self.n_slots = 84 if self.n_processes > 1 else 64

    def __repr__(self):
        return "SDDP"

    def header(self):
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}}".format("SDDP Solver, Zhen Chen", width = self.n_slots)
        )
        self.logger.info("-" * self.n_slots)
        if self.n_processes > 1:
            self.logger.info(
                "{:>12s}{:>20s}{:^40s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value {}% CI ({})".format(self.percentile, self.n_processes),
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


if __name__ == '__main__':
    test = Logger()
