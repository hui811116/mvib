import numpy as np
import sys
import argparse 
import time
import os
import pickle
import mvdataset as da
#import mvalg as alg
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


gamma_vec=  np.array([0.2,0.2])
penalty = 48.0

sel_data = da.select("syn2_simple")

pxy_list = sel_data['pxy_list']
tf_pxy_list = [tf.convert_to_tensor(item,dtype=tf.float32) for item in pxy_list]
nview = len(pxy_list)

# testing mvcc
convthres = 1e-5
maxiter = 20000
ss_init = 1e-3
ss_scale = 0.25
retry_num = 10
sys_param = {'ss_init':ss_init,'ss_scale':ss_scale,'retry':retry_num,'penalty':penalty}
# NOTE: first-round debugging complete
mv_outdict = alg.tf_mvib_cc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)
# NOTE: first-round debugging complete
#ba_out= alg.tf_ib_orig(tf_pxy_list[1],gamma_vec[1],convthres,maxiter,**sys_param)
# NOTE: first-round debugging complete
#admmib_out = alg.tf_admmib_type1(tf_pxy_list[0],gamma_vec[0],convthres,maxiter,**sys_param)
# NOTE: first-round debugging complete
inc_outdict=  alg.tf_mvib_inc(tf_pxy_list,gamma_vec,convthres,maxiter,**sys_param)





