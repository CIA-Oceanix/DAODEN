"""Logging format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
from datetime import datetime
import socket
import os

def new_log(logdir,filename):
    filename = os.path.join(logdir,
                            datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S_"+socket.gethostname()+"_"+filename+".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
                        