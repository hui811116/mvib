import numpy as np
import json
import sys
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import argparse
import pprint

d_base = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('data',type=str,help='directory to data')
parser.add_argument('-param',type=str,help='the parameter set generating the data',default='')

argsdict = vars(parser.parse_args())

with open(os.path.join(d_base,argsdict['data']),'rb') as fid:
	data_all = np.load(fid)
'''
res_np[gi1,gi2,ni,:] = np.array([ga1,ga2,tmp_pzcxlist[0],tmp_pzcxlist[1],tmp_mizy,int(algout['conv']),algout['niter']])
'''

#print(data_all.shape)
param_dict = {}
if argsdict['param']!='':
	with open(os.path.join(d_base,argsdict['param']),'r') as fj:
		param_dict = json.load(fj)
#print(param_dict)
'''
{'output': 'ssinit1em2_c8', 'thres': 1e-05, 'seed': None, 
'maxiter': 50000, 'penalty': 8.0, 'ss_init': 0.01, 'ss_scale': 0.25, 
'dataset': 'syn2', 'gamma_min': 0.01, 'gamma_max': 0.5, 'ngamma': 20, 
'niter': 25, 'verbose': 1, 'penalty_coefficient': 8.0, 'init_stepsize': 0.01, 
'stepsize_scale': 0.25, 'armijo_c1': 0.0001, 'rand_seed': None}
'''
ds = data_all.shape
data_2d = data_all.reshape((np.prod(ds[:-1]),ds[-1]))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_2d[:,2],data_2d[:,3],data_2d[:,4])
ax.set_xlabel(r'$I(Z;X^{(1)})$ (nats)')
ax.set_ylabel(r'$I(Z;X^{(2)})$ (nats)')
ax.set_zlabel(r'$I(Z;Y)$ (nats)')
plt.show()


fig= plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_2d[:,0],data_2d[:,1],100*data_2d[:,5])
ax.set_xlabel(r'$\gamma_1$')
ax.set_ylabel(r'$\gamma_2$')
ax.set_zlabel(r'Convergent cases (\%)')
plt.show()