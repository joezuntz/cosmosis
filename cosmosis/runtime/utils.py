import numpy as np
import argparse
import os
import errno
from timeit import default_timer

class EverythingIsNan(object):
    def __getitem__(self, param):
        return np.nan

everythingIsNan = EverythingIsNan()

class ParseExtraParameters(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if getattr(args, self.dest, self.default) is not None:
            parser.error(option_string + " appears several times")
        result = {}
        for arg in values:
            section, param_value = arg.split('.',1)
            param,value = param_value.split('=',1)
            result[(section,param)] = value
        setattr(args, self.dest, result)

def mkdir(path):
    #This is much nicer in python 3.
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if os.path.isdir(path):
                #error is that dir already exists; fine - no error
                pass
            else:
                #error is that file with name of dir exists already
                raise ValueError("Tried to create dir %s but file with name exists already"%path)
        elif error.errno == errno.ENOTDIR:
            #some part of the path (not the end) already exists as a file 
            raise ValueError("Tried to create dir %s but some part of the path already exists as a file"%path)
        else:
            #Some other kind of error making directory
            raise


class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        interval = default_timer() - self.start
        print "Time taken by step '{}': {}".format(self.msg, interval)

def symmetrized_matrix(U):
    M = U.copy()
    inds = np.triu_indices_from(M,k=1)
    M[(inds[1], inds[0])] = M[inds]
    return M


def symmetric_positive_definite_inverse(M):
    import scipy.linalg
    U,status = scipy.linalg.lapack.dpotrf(M)
    if status != 0:
        raise ValueError("Non-symmetric positive definite matrix")
    M,status = scipy.linalg.lapack.dpotri(U)
    if status != 0:
        raise ValueError("Error in Cholesky factorization")
    M = symmetrized_matrix(M)
    return M
