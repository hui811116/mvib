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
import mytest as mt
import copy

# this demo consider a simple two-view scenario

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('method',type=str,choices=alg.availableAlgs(),help="select the method")
parser.add_argument('-thres',type=float,help='Convergence threshold',default=1e-6)
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=50000)
parser.add_argument('-penalty',type=float,help='The Penalty Coefficient',default=64.0)
parser.add_argument('-ss_init',type=float,help='Initial value for step size search',default=1e-2)
parser.add_argument('-ss_scale',type=float,help='Scaling value for step size search',default=0.25)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2_simple")
parser.add_argument('-gamma',type=str,help='Gammas of each view, format:0.1,0.2,...',default="0.08,0.08")
parser.add_argument('-niter',type=int,help='Number of iterations per gamma vectors',default=25)
parser.add_argument('-retry',type=int,help='Retry for each internal solver',default=10)

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
	'retry'    :           argsdict['retry'],
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
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}
algsel = alg.select(argsdict['method'])
# FIXME: debugging, no output for now
mv_outdict = algsel(**{**alg_args,**sys_param})


# preparing the testing data

# by conditional independence
xdim_list = [item.shape[0] for item in pxy_list]
rcnt = 0
idx_map ={}
back_idx_map = {}
nview = len(pxy_list)
start_arr = np.zeros((nview),dtype=int)
end_flag = False
while not end_flag:
	idx_map[np.array2string(start_arr)] = {'index':rcnt,'array':copy.copy(start_arr)}
	back_idx_map[rcnt] = {'array':copy.copy(start_arr)}
	rcnt+=1
	start_arr[-1] += 1
	for ci in range(nview):
		if start_arr[-ci-1] == xdim_list[-ci-1]:
			if ci == nview-1:
				end_flag = True
				break
			start_arr[-ci-1] = 0
			start_arr[-ci-2]+= 1

# combining the two views
pxy_merge = np.zeros((np.prod([item.shape[0] for item in pxy_list]),py.shape[0]))
for ix in range(pxy_merge.shape[0]):
	for iy in range(pxy_merge.shape[1]):
		sep_idx = back_idx_map[ix]['array']
		tmp_sum = 0
		tmp_ele = []
		for iit, item in enumerate(sep_idx):
			tmp_ele.append((pxy_list[iit]/py[None,:])[sep_idx[iit],iy])
		pxy_merge[ix,iy] = np.prod(np.array(tmp_ele))
pxy_merge *= py[None,:]

# Now we should prepare a testbed for all kinds of multiview/single-view methods
# the testing dataset should be fixed...
# the testing data are in pickle format
#{x_test:[num,nview], y_test:[num,label]}
if argsdict['dataset'] == 'syn2_simple':
	load_testdata = 'testdata_syn2_simple_num_10000.pkl'
elif argsdict['dataset'] == 'syn2_inc':
	load_testdata = 'testdata_syn2_inc_num_20000.pkl'
else:
	sys.exit('testing dataset not found')

with open(load_testdata,'rb') as fid:
	data_test = pickle.load(fid)
x_test_nv = data_test['x_test'].astype(int)
y_test   = data_test['y_test'].astype(int)

# for multiview decoders

if mv_outdict['conv']:
	# a convergent case
	# consider equal weight
	consensus_weight = 0.9
	gamma_max = np.amax(gamma_vec)
	if argsdict['method'] == "cc":
		'''
		mizy_all = np.array(mv_outdict['IYZ_list'])+np.array(mv_outdict['IYZC_list'])
		print('sum IZY:',mizy_all)
		mvcc_test = mt.mvCcTest(mv_outdict['con_enc'],mv_outdict['cmpl_enc'][0],pxy_list,**{**argsdict,'view_idx':0,'consensus_weight':consensus_weight,})
		y_cc_est = mvcc_test.test(x_test_nv,y_test)
		#y_cc_est = mvcc_test.test_z(x_test_nv[:,0],y_test)
		#y_cc_est = mvcc_test.test_c(x_test_nv[:,0],y_test)
		test_acc = np.sum(np.equal(y_test,y_cc_est))/y_test.shape[0]
		print('MVCC-view1: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[0],test_acc*100))
		mvcc_test_2 = mt.mvCcTest(mv_outdict['con_enc'],mv_outdict['cmpl_enc'][1],pxy_list,**{**argsdict,'view_idx':1,'consensus_weight':consensus_weight,})
		y_cc_est_2 = mvcc_test_2.test(x_test_nv,y_test)
		#y_cc_est_2 = mvcc_test_2.test_z(x_test_nv[:,1],y_test)
		#y_cc_est_2 = mvcc_test_2.test_c(x_test_nv[:,1],y_test)
		test_acc_2 = np.sum(np.equal(y_test,y_cc_est_2))/y_test.shape[0]
		print('MVCC-view2: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[1],test_acc_2*100))
		'''
		mvcc_test = mt.mvCcFullTest(mv_outdict['con_enc'],mv_outdict['cmpl_enc'],pxy_list,**argsdict)
		y_cc_test = mvcc_test.test(x_test_nv,y_test)
		test_acc = np.sum(np.equal(y_cc_test,y_test))/y_test.shape[0]
		print('MV-CC: gamma_1={:>6.3f}, gamma_2={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[0],gamma_vec[1],test_acc*100))
	elif argsdict['method'] == "inc":
		mvinc_test = mt.mvIncTestTwo(mv_outdict['enc_list'],mv_outdict['dec_list'],pxy_list,**argsdict)
		y_inc_est = mvinc_test.test(x_test_nv,y_test)
		test_acc = np.sum(np.equal(y_test,y_inc_est))/y_test.shape[0]
		print('MV-INC: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_max,test_acc*100))


# merging baseline
x_mergetwo_test = np.zeros(x_test_nv.shape[0],dtype=int)
for nn in range(x_test_nv.shape[0]):
	x_mergetwo_test[nn] = idx_map[np.array2string(x_test_nv[nn,:])]['index']

# Now the goal is to set a unified testbed
# each model predict things in its own way...

ba_merge_out = alg.ib_orig(**{'pxy':pxy_merge,'gamma':gamma_vec[0],**alg_args,**sys_param})
merge_enc = ba_merge_out['prob_zcx']
ba_mergetwo_test = mt.singleViewTest(merge_enc,pxy_merge,**argsdict)
y_bamergetwo_est = ba_mergetwo_test.test(x_mergetwo_test,y_test)
ba_mergetwo_acc =np.sum((y_test == y_bamergetwo_est))/y_test.shape[0]
print('BA-MergeTwo: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[0],ba_mergetwo_acc *100))


# single-view baseline
ba_v1_out = alg.ib_orig(**{'pxy':pxy_list[0],'gamma':gamma_vec[0],**alg_args,**sys_param})
v1_enc = ba_v1_out['prob_zcx']
bav1_test = mt.singleViewTest(v1_enc,pxy_list[0],**argsdict)
y_bav1_est = bav1_test.test(x_test_nv[:,0],y_test)
ba_v1_acc = np.sum(np.equal(y_bav1_est,y_test))/y_test.shape[0]
print('BA-View 1: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[0],ba_v1_acc * 100))


ba_v2_out = alg.ib_orig(**{'pxy':pxy_list[1], 'gamma':gamma_vec[1], **alg_args,**sys_param})
v2_enc = ba_v2_out['prob_zcx']
bav2_test = mt.singleViewTest(v2_enc,pxy_list[1],**argsdict)
y_bav2_est = bav2_test.test(x_test_nv[:,1],y_test)
ba_v2_acc = np.sum(np.equal(y_bav2_est,y_test))/y_test.shape[0]
print('BA-View 2: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_vec[0],ba_v2_acc*100))


