import logging

filename = "{}.log".format(__file__)
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG,
    filename=filename,
    filemode="w",
    format=fmt
)

logging.info("info")
logging.debug("debug")
logging.warning("warning")
logging.error("error")
logging.critical("critical")