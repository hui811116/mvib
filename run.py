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
import json

# this demo consider a simple two-view scenario

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('method',type=str,choices=alg.availableAlgs(),help="select the method")
parser.add_argument('output',type=str,help='results filename')
parser.add_argument('-thres',type=float,help='Convergence threshold',default=1e-5)
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=50000)
parser.add_argument('-penalty',type=float,help='The Penalty Coefficient',default=8.0)
parser.add_argument('-ss_init',type=float,help='Initial value for step size search',default=1e-2)
parser.add_argument('-ss_scale',type=float,help='Scaling value for step size search',default=0.25)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2")
parser.add_argument('-gamma_min',type=float,help="the minimum gamma for grid search",default=0.05)
parser.add_argument('-gamma_max',type=float,help="the maximum gamma for grid search",default=0.66)
parser.add_argument('-ngamma',type=int,help="Spacing of the gamma grid",default=20)
parser.add_argument('-niter',type=int,help='Number of iterations per gamma vectors',default=25)
parser.add_argument('-v',"--verbose",help='Printing the log and parameters along the execution',action='count',default=0)
#parser.add_argument('-p',"--parallel",help='Solve with multiprocessing unit',action='count',default=0)

# MACRO for Developing
argsdict = vars(parser.parse_args())
data_dict = da.select(argsdict['dataset'])

gamma_min = argsdict['gamma_min']
gamma_max = argsdict['gamma_max']
gamma_num = argsdict['ngamma']
ga_axis1 = np.geomspace(gamma_min,gamma_max,gamma_num)
ga_axis2 = np.geomspace(gamma_min,gamma_max,gamma_num)

# system parameter integration
sys_param = {
	'penalty_coefficient': argsdict['penalty'],
	'init_stepsize':       argsdict['ss_init'],
	'stepsize_scale':      argsdict['ss_scale'],
	'armijo_c1':           1e-4,
	'rand_seed':           argsdict['seed'],
}

if argsdict.get('verbose',False):
	pprint.pprint(sys_param)

mvib_alg = alg.select(argsdict['method'])

# running range
pxy_list  = data_dict['pxy_list']
mixy_list = [ut.calcMI(item) for item in pxy_list]
px_list   = [np.sum(item,axis=1) for item in pxy_list]
py_list   = [np.sum(item,axis=0) for item in pxy_list] #NOTE: right now, py is assumed to be the same over all views
py        = py_list[0]
alg_args  = {
	'pxy_list' : pxy_list,
	'nz':data_dict['nz'],
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}

run_start = time.time()
spformat = '{{:<{}}}'.format(os.get_terminal_size().columns)
niter = argsdict['niter']
# results
res_np = np.zeros((len(ga_axis1),len(ga_axis2),niter,7))
# zeros...[gamma1,gamma2,mizx1,mizx2,mizy,conv,niter]
for gi1,ga1 in enumerate(ga_axis1):
	for gi2,ga2 in enumerate(ga_axis2):
		gamma_vec = np.array([ga1,ga2])
		iter_ts = time.time()
		cnt_conv = 0
		for ni in range(niter):
			pstring = 'Current Progress: gamma1={:5.3f},gamma2={:5.3f}---iteration({:>5d}/{:>5d})---time elapsed: {:>16.4f} seconds'.format(ga1,ga2,ni,niter,time.time()-run_start)
			print(spformat.format(pstring),end='\r',flush=True)
			#algout = alg.mvib_nv(**{**alg_args,**sys_param,'gamma_vec':gamma_vec})
			#algout = alg.mvib_nv_parallel(**{**alg_args,**sys_param,'gamma_vec':gamma_vec})
			algout = mvib_alg(**{**alg_args,**sys_param,'gamma_vec':gamma_vec})
			cnt_conv += int(algout['conv'])
			tmp_pzcxlist = [ut.calcMI(item * px_list[idx][None,:]) for idx,item in enumerate(algout['pzcx_list'])]
			tmp_mizy = ut.calcMI(algout['pzcy']*py[None,:])
			res_np[gi1,gi2,ni,:] = np.array([ga1,ga2,tmp_pzcxlist[0],tmp_pzcxlist[1],tmp_mizy,int(algout['conv']),algout['niter']])
		pstring = 'gamma1={:5.3f},gamma2={:5.3f}---complete. Avg. sec/iter: {:>8.4f}, Conv. rate: {:>8.4f}'.format(ga1,ga2,(time.time()-iter_ts)/niter,cnt_conv/niter)
		print(spformat.format(pstring),flush=True)
print('Simulation complete. total time elapsed: {:>16.4f} seconds'.format(time.time()-run_start))

# saving the results...
filename= argsdict['output']
idx = 0
while os.path.isfile(os.path.join(d_base,filename+'.npy')):
	# the filename exists
	filename += str(idx+1)
	idx+=1
save_file_dir = os.path.join(d_base,filename+'.npy')
with open(save_file_dir,'wb') as fid:
	print('Saving the results to: {:}'.format(save_file_dir))
	np.save(fid,res_np)

# saving the parameters
with open(os.path.join(d_base,filename+'.json'),'w') as fj:
	json.dump({**argsdict,**sys_param},fj)

'''
algout = alg.mvib_nv(**{**alg_args,**sys_param})
# calculate mizx for each view
mizx_list = [ut.calcMI(algout['pzcx_list'][idx] * px_list[idx][None,:]) for idx in range(len(pxy_list))]
mizy = ut.calcMI(algout['pzcy'] *py[None,:])

pprint.pprint({'IXY_list':mixy_list,'IZX_list':mizx_list,'IZY':mizy,'converged':algout['conv'],'num_iter':algout['niter']})
'''