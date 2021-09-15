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

# this demo consider a simple two-view scenario

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('-thres',type=float,help='Convergence threshold',default=1e-4)
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=10000)
parser.add_argument('-penalty',type=float,help='The Penalty Coefficient',default=54.0)
parser.add_argument('-ss_init',type=float,help='Initial value for step size search',default=0.01)
parser.add_argument('-ss_scale',type=float,help='Scaling value for step size search',default=0.20)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2")
parser.add_argument('-gamma',type=str,help='Gammas of each view, format:0.1,0.2,...',default="0.3,0.5")
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
pxy_list = data_dict['pxy_list']
alg_args = {
	'pxy_list' : pxy_list,
	'gamma_vec': gamma_vec,
	'nz':data_dict['nz'],
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}
#algout = alg.mvib_2v(**{**alg_args,**sys_param})
algout = alg.mvib_nv(**{**alg_args,**sys_param})
print(algout)