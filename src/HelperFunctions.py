import sys
import logging
import os
import numpy as np
import theano.tensor as T

def endSound():
    duration = 0.2  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq*1.5))

def endSound2():
    duration = 0.2  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq*1.5))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def setUpLogger(fileName):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(fileName, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger("pymc3")
    logger.handlers = []

def round_sig(f, p):
    return float(('%.' + str(p) + 'e') % f)

def getIndex(array, compareStr):
    for count, item in enumerate(array):
        if compareStr in item:
            return count
    raise Exception(f"The string: {compareStr} is not contained in the array")

def all_subdirs_of(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result

def chisquare(obs,exp):
    return np.sum(obs**2/exp**2)/obs.shape[0]