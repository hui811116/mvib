import numpy as np
import sys
import argparse 
import time
import os
import pickle
import mvdataset as da
import mvalg as alg
import mvgrad as gd
import mvutils as ut
import matplotlib.pyplot as plt
import pprint

# this demo consider a simple two-view scenario

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('method',type=str,choices=alg.availableAlgs(),help="select the method")
parser.add_argument('-thres',type=float,help='Convergence threshold',default=1e-5)
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=50000)
parser.add_argument('-penalty',type=float,help='The Penalty Coefficient',default=24.0)
parser.add_argument('-ss_init',type=float,help='Initial value for step size search',default=5e-3)
parser.add_argument('-ss_scale',type=float,help='Scaling value for step size search',default=0.25)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2_simple")
parser.add_argument('-gamma',type=str,help='Gammas of each view, format:0.1,0.2,...',default="0.05,0.05")
parser.add_argument('-niter',type=int,help='Number of iterations per gamma vectors',default=25)

# MACRO for Developing
argsdict = vars(parser.parse_args())
data_dict = da.select(argsdict['dataset'])
gamma_vec = np.array(argsdict['gamma'].split(',')).astype('float32')

assert len(gamma_vec) == len(data_dict['pxy_list'])

# system parameter integration
sys_param = {
	'penalty_coefficient': argsdict['penalty'],
	'init_stepsize':       argsdict['ss_init'],
	'stepsize_scale':      argsdict['ss_scale'],
	'armijo_c1':           1e-4,
	'rand_seed':           argsdict['seed'],
}

# running range
pxy_list  = data_dict['pxy_list']
mixy_list = [ut.calcMI(item) for item in pxy_list]
px_list   = [np.sum(item,axis=1) for item in pxy_list]
py_list   = [np.sum(item,axis=0) for item in pxy_list] #NOTE: right now, py is assumed to be the same over all views
py        = py_list[0]
alg_args  = {
	'pxy_list' : pxy_list,
	'gamma_vec': gamma_vec,
	'nz':len(py),
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}
algsel = alg.select(argsdict['method'])
# FIXME: debugging, no output for now
algsel(**{**alg_args,**sys_param})


# Now we should prepare a testbed for all kinds of multiview/single-view methods
# the testing dataset should be fixed...

# the testing data are in pickle format
#{x_test:[num,nview], y_test:[num,label]} 

# Now the goal is to set a unified testbed
# each model predict things in its own way...