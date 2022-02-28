#!/usr/bin/python
# -*- encoding: utf-8 -*-
import time
import logging
import os.path as osp


def setup_logger(logpth):
    log_level = logging.INFO
    format = '%(asctime)s %(levelname)s(%(lineno)d): %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    # logfile = 'AttaNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = 'AttaNet.log'
    logfile = osp.join(logpth, logfile)

    logging.basicConfig(level=log_level, format=format, datefmt=datefmt, filename=logfile)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console)
    logger = logging.getLogger()
    return logger

