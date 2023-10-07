import logging
import logging.handlers

def setup_logger(logger_name):
    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level to DEBUG to capture all messages
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logfile.log'),  # Log messages to a file
            logging.StreamHandler()  # Log messages to the console
        ]
    )
    return logging.getLogger(logger_name)  # Return the logger object with the specified name