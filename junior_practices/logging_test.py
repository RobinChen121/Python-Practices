"""
Created on 2025/1/17, 23:00 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""

import logging

# Configure logging
logging.basicConfig(
    level = logging.DEBUG,  # Set the minimum level of messages to log
    format = '%(asctime)s -	%(lineno)d- %(levelname)s - %(message)s',  # Specify the format
    filename = 'app.log',  # Log to a file
    filemode = 'w'  # Overwrite the log file on each run
)

# Example log messages
logging.debug('This is a debug message.')
logging.info('This is an info message.')
logging.warning('This is a warning message.')
logging.error('This is an error message.')
logging.critical('This is a critical message.')
