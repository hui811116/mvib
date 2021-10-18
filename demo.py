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
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=20000)
parser.add_argument('-penalty',type=float,help='The Penalty Coefficient',default=4.0)
parser.add_argument('-ss_init',type=float,help='Initial value for step size search',default=1e-2)
parser.add_argument('-ss_scale',type=float,help='Scaling value for step size search',default=0.125)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2")
parser.add_argument('-gamma',type=str,help='Gammas of each view, format:0.1,0.2,...',default="0.08,0.05")
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
	'nz':data_dict['nz'],
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}
algsel = alg.select(argsdict['method'])
if argsdict['method'] != "complement":
	algout = algsel(**{**alg_args,**sys_param})
else:
	cmpl_param = {
		"gamma_cmpl": gamma_vec[0],
		"nzc": data_dict['nz'],
		#"nze_vec":np.array([data_dict['nz']]*len(pxy_list)),   #FIXME: right now the same as nzc for basic result guarantee
		"nze_vec":np.array([item.shape[0] for item in pxy_list]),
	}
	algout = algsel(**{**alg_args,**sys_param,**cmpl_param})
# calculate mizx for each view
mizx_list = [ut.calcMI(algout['pzcx_list'][idx] * px_list[idx][None,:]) for idx in range(len(pxy_list))]
mizy = ut.calcMI(algout['pzcy'] *py[None,:])

pprint.pprint({'IXY_list':mixy_list,'IZX_list':mizx_list,'IZY':mizy,'converged':algout['conv'],'num_iter':algout['niter']})