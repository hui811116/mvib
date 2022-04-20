import numpy as np
import sys
import argparse 
import time
import os
import pickle
import mvdataset as da
import mvalg as npalg
import mvgrad as gd
import mvutils as ut
import matplotlib.pyplot as plt
import pprint
import mytest as mt
import copy
from scipy.io import savemat
import itertools
import tensorflow as tf
from tensorflow import keras
import mv_tf_alg as alg


#gamma_vec=  np.array([0.1,0.1])
penalty = 64.0

# loading mnist_2v
with open('mnist_vib_v1_pred.npy','rb') as fid:
	mnist_v1_pycx = np.load(fid)
with open('mnist_vib_v2_pred.npy','rb') as fid:
	mnist_v2_pycx = np.load(fid)
px_v1 = np.ones(mnist_v1_pycx.shape[1])*(1/mnist_v1_pycx.shape[1])
px_v2 = np.ones(mnist_v2_pycx.shape[1])*(1/mnist_v2_pycx.shape[1])
pxy_list = [(mnist_v1_pycx*px_v1[None,:]).T , (mnist_v2_pycx*px_v2[None,:]).T]
#pxy_list = [(mnist_v2_pycx*px_v2[None,:]).T , (mnist_v1_pycx*px_v1[None,:]).T]

#print([item.shape for item in pxy_list])
#sys.exit()
#sel_data = da.select("syn2_simple")
#sel_data = da.select("syn2_inc")
#pxy_list = sel_data['pxy_list']
tf_pxy_list = [tf.convert_to_tensor(item,dtype=tf.float32) for item in pxy_list]
nview = len(pxy_list)

# testing mvcc
convthres = 1e-5
maxiter = 30000
ss_init = 5e-3
ss_scale = 0.25
retry_num = 10
sys_param = {'ss_init':ss_init,'ss_scale':ss_scale,'retry':retry_num,'penalty':penalty,
			'rand_seed':None,'penalty_coefficient':penalty,
			'init_stepsize':ss_init,'stepsize_scale':ss_scale}

# debugging for the consensus step
#np_mvnv_out = npalg.mvib_nv(pxy_list,gamma_vec,convthres,maxiter,**sys_param)
#print(np_mvnv_out['conv'],np_mvnv_out['niter'],np_mvnv_out['IXZ_list'],np_mvnv_out['IYZ_list'])
#tf_mvnv_out = alg.tf_mvcc_nv(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
#print(tf_mvnv_out['conv'],tf_mvnv_out['niter'],tf_mvnv_out['IXZ_list'],tf_mvnv_out['IYZ_list'])

'''
mv_outdict = alg.tf_mvib_cc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
if mv_outdict['conv']:
	print(mv_outdict['conv'],mv_outdict['IXZC_list'],mv_outdict['IYZC_list'])
else:
	print('mv divergent')
'''

# NOTE: first-round debugging complete
#ba_out= alg.tf_ib_orig(tf_pxy_list[1],gamma_vec[1],convthres,maxiter,**sys_param)
#print(ba_out['conv'],ba_out['IXZ'],ba_out['IYZ'],ba_out['niter'])
# NOTE: first-round debugging complete
#admmib_out = alg.tf_admmib_type1(tf_pxy_list[0],gamma_vec[0],convthres,maxiter,**sys_param)
#print(admmib_out['conv'],admmib_out['IZX'],admmib_out['IZY'],admmib_out['niter'])
# NOTE: first-round debugging complete

'''
inc_outdict=  alg.tf_mvib_inc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
if inc_outdict['conv']:
	print(inc_outdict['conv'],inc_outdict['IXZ_list'],inc_outdict['IYZ_list'])
else:
	print('inc divergent')
'''

''
gamma_range = np.arange(0.1,0.9,0.1)
niter = 25
'''
res_all = np.zeros((len(gamma_range)*niter,7)) # gamma,penalty,IZX,IXZC,IZY,IYZC,conv
rcnt = 0
for gamma in gamma_range:
	gamma_vec= np.array([gamma,gamma])
	for nn in range(niter):
		inc_outdict = alg.tf_mvib_inc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
		if inc_outdict['conv']:
			mizx_list = inc_outdict['IXZ_list']
			mizy_list = inc_outdict['IYZ_list']
			res_all[rcnt,:] = [gamma,penalty,mizx_list[0],mizx_list[1],mizy_list[0],mizy_list[1],1]
		else:
			res_all[rcnt,:]= [gamma,penalty,0,0,0,0,0]
		rcnt+=1
#savemat("mnist_2vrev_inc.mat",{'data':res_all,'label':'mnist_2v_inc'})
savemat("debug_syn2simple_inc.mat",{'data':res_all,'label':'syn2simple_inc'})
'''
np_all = np.zeros((len(gamma_range)*niter,7)) 
rcnt = 0
for gamma in gamma_range:
	gamma_vec= np.array([gamma,gamma])
	for nn in range(niter):
		#inc_outdict = alg.tf_mvib_inc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
		npinc_outdict = npalg.mvib_inc(pxy_list,gamma_vec,convthres,maxiter,**sys_param)
		if npinc_outdict['conv']:
			mizx_list = npinc_outdict['IXZ_list']
			mizy_list = npinc_outdict['IYZ_list']
			np_all[rcnt,:] = [gamma,penalty,mizx_list[0],mizx_list[1],mizy_list[0],mizy_list[1],1]
		else:
			np_all[rcnt,:]= [gamma,penalty,0,0,0,0,0]
		rcnt+=1
savemat("np_mnist_inc.mat",{'data':np_all,'label':'np_mnist_inc'})

'''
# for ba, only run single view for each... otherwise it is infeasible
gamma_range = np.arange(0.1,0.9,0.1)
niter =25
res_ba = np.zeros((len(gamma_range)*niter,4)) # gamma,IZX,IZY,conv
rcnt = 0
for gamma in gamma_range:
	for nn in range(niter):
		ba_outdict=  alg.tf_ib_orig(tf_pxy_list[0],gamma,convthres,maxiter,**sys_param)
		if ba_outdict['conv']:
			res_ba[rcnt,:] = [gamma, ba_outdict['IXZ'],ba_outdict['IYZ'],1]
		else:
			res_ba[rcnt,:] = [gamma, 0,0,0]
		rcnt+=1
savemat("mnist_2v_ba_v0.mat",{'data':res_ba,'label':'mnist_2v_ba_v0'})

res_ba2 = np.zeros((len(gamma_range)*niter,4)) # same
rcnt = 0
for gamma in gamma_range:
	for nn in range(niter):
		ba_outdict = alg.tf_ib_orig(tf_pxy_list[1],gamma,convthres,maxiter,**sys_param)
		if ba_outdict['conv']:
			res_ba2[rcnt,:] = [gamma, ba_outdict['IXZ'],ba_outdict['IYZ'],1]
		else:
			res_ba2[rcnt,:] = [gamma,0,0,0]
		rcnt+=1
savemat("mnist_2v_ba_v1.mat",{'data':res_ba2,'label':'mnist_2v_ba_v1'})
'''