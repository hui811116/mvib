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


gamma_vec=  np.array([0.1,0.1])
penalty = 48.0

# loading mnist_2v
with open('mnist_vib_v1_pred.npy','rb') as fid:
	mnist_v1_pycx = np.load(fid)
with open('mnist_vib_v2_pred.npy','rb') as fid:
	mnist_v2_pycx = np.load(fid)
px_v1 = np.ones(mnist_v1_pycx.shape[1])*(1/mnist_v1_pycx.shape[1])
px_v2 = np.ones(mnist_v2_pycx.shape[1])*(1/mnist_v2_pycx.shape[1])
pxy_list = [(mnist_v1_pycx*px_v1[None,:]).T , (mnist_v2_pycx*px_v2[None,:]).T]

#sel_data = da.select("syn2_simple")
#pxy_list = sel_data['pxy_list']
tf_pxy_list = [tf.convert_to_tensor(item,dtype=tf.float32) for item in pxy_list]
nview = len(pxy_list)

# testing mvcc
convthres = 1e-5
maxiter = 50000
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


mv_outdict = alg.tf_mvib_cc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
if mv_outdict['conv']:
	print(mv_outdict['conv'],mv_outdict['IXZC_list'],mv_outdict['IYZC_list'])
else:
	print('mv divergent')

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