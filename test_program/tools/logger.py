import logging
from logging import handlers 

class Logger(object):
    def __init__(self, log_path, level='info', 
    fmt='%(asctime)s -%(pathname)s[line:%(lineno)d]- %(levelname)s: %(message)s '):
        super().__init__()
        self.logger = logging.getLogger(log_path)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.INFO)

        sh = logging.StreamHandler() # screen output
        sh.setFormatter(format_str)

        th = handlers.TimedRotatingFileHandler(filename=log_path, encoding='utf-8', when='D')

        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('./test.log', level='info')
    log.logger.info(f'nke test ')