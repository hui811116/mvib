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
from scipy.io import savemat
import itertools

# this demo consider a simple two-view scenario

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('method',type=str,choices=alg.availableAlgs(),help="select the method")
parser.add_argument('-thres',type=float,help='Convergence threshold',default=1e-6)
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='Maximum number of iteration',default=50000)
parser.add_argument('-ssinit',type=float,help='Initial value for step size search',default=5e-3)
parser.add_argument('-sscale',type=float,help='Scaling value for step size search',default=0.25)
parser.add_argument('-dataset',type=str,help='The dataset for simulation',default="syn2_simple")
parser.add_argument('-gamma_min',type=float,help="the minimum gamma for grid search",default=0.05)
parser.add_argument('-gamma_max',type=float,help="the maximum gamma for grid search",default=1.0)
parser.add_argument('-ngamma',type=int,help="Spacing of the gamma grid",default=16)
parser.add_argument('-niter',type=int,help='Number of iterations per gamma vectors',default=50)
parser.add_argument('-retry',type=int,help='Retry for each internal solver',default=10)

# MACRO for Developing
argsdict = vars(parser.parse_args())
data_dict = da.select(argsdict['dataset'])

gamma_min = argsdict['gamma_min']
gamma_max = argsdict['gamma_max']
gamma_num = argsdict['ngamma']

penalty_min = 32.0
penalty_max = 96.0
penalty_step = 32.0

agg_on = False

gamma_range = np.geomspace(gamma_min,gamma_max,gamma_num)
penalty_range =np.arange(penalty_min,penalty_max+1,penalty_step)
# system parameter integration
sys_param = {
	#'penalty_coefficient': argsdict['penalty'],
	'init_stepsize':       argsdict['ssinit'],
	'stepsize_scale':      argsdict['sscale'],
	#'armijo_c1':           1e-4,
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
	#'gamma_vec': gamma_vec,
	'convthres':argsdict['thres'],
	'maxiter':argsdict['maxiter'],
}


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

# merging baseline
x_mergetwo_test = np.zeros(x_test_nv.shape[0],dtype=int)
for nn in range(x_test_nv.shape[0]):
	x_mergetwo_test[nn] = idx_map[np.array2string(x_test_nv[nn,:])]['index']


if agg_on:
	# manually open compared scheme
	# we get the joint pxy_merge
	# py
	pxcy = pxy_merge / py[None,:]
	agg_pzcx = np.eye(pxy_merge.shape[0])
	shf_idx = np.random.permutation(pxcy.shape[0])
	cpy_pzcx = np.zeros((pxcy.shape[0],pxcy.shape[0]))
	for idx, si in enumerate(shf_idx):
		cpy_pzcx[:,idx] = agg_pzcx[:,si]
	agg_pzcx = cpy_pzcx
	#print(agg_pzcx)
	curmi = ut.calcMI(agg_pzcx@pxy_merge)
	while agg_pzcx.shape[0]>2:
		idx_range = np.arange(0,agg_pzcx.shape[0])
		combs = list(itertools.combinations(idx_range,2))
		best_diff = ut.calcMI(agg_pzcx @ pxy_merge)
		best_pzcx = None
		for merge_idx in combs:
			np_merge_idx = np.array(list(merge_idx))
			all_idx = np.ones((agg_pzcx.shape[0]))
			all_idx[np_merge_idx] = 0
			next_pzcx = np.zeros((agg_pzcx.shape[0]-1,pxy_merge.shape[0]))
			rcnt =0
			for ic in range(agg_pzcx.shape[0]):
				if all_idx[ic] != 0:
					next_pzcx[rcnt,:] = agg_pzcx[ic,:]
					rcnt +=1
			next_pzcx[-1,:]= np.sum(agg_pzcx[np_merge_idx,:],axis=0)
			next_pzcy = next_pzcx @ pxcy
			midiff = curmi - ut.calcMI(next_pzcy*py[None,:])
			if midiff < best_diff:
				best_diff = midiff
				best_pzcx = next_pzcx
		agg_pzcx = best_pzcx
		tmp_agg_test = mt.singleViewTest(agg_pzcx,pxy_merge,**argsdict)
		y_agg_est = tmp_agg_test.test(x_mergetwo_test,y_test)
		agg_acc = np.sum((y_agg_est == y_test))/y_test.shape[0]
		print('Agg testing acc: {:>6.3f}, z={:>3}'.format(agg_acc*100,agg_pzcx.shape[0]))

	sys.exit()
# helper function
def saveBaResult(file_template,res_data,hdr,**kwargs):
	fcnt = 0
	fname = file_template.format(kwargs['dataset'],fcnt)
	savefile = os.path.join(d_base,fname)
	while os.path.isfile(savefile):
		fcnt+=1
		fname = file_template.format(kwargs['dataset'],fcnt)
		savefile = os.path.join(d_base,fname)
	savemat(savefile,{'data':res_data,'hdr':hdr})
# MAIN algorithm

if argsdict['method'] == "ba":
	# run view 1 and view 2 + merge view
	res_hdr ="gamma,IZX,IZY,niter,test_acc,conv"
	ba_merge_res = np.zeros((argsdict['niter']*gamma_range.shape[0],6)) #[gamma,IZX,IZY,niter,test_acc,conv]
	ba_view1_res = np.zeros((argsdict['niter']*gamma_range.shape[0],6))
	ba_view2_res = np.zeros((argsdict['niter']*gamma_range.shape[0],6))
	itcnt = 0
	for nn in range(argsdict['niter']):
		for gamma in gamma_range:
			# merge view

			ba_merge_out = alg.ib_orig(**{'pxy':pxy_merge,'gamma':gamma,**alg_args,**sys_param})
			merge_enc = ba_merge_out['prob_zcx']
			ba_mergetwo_test = mt.singleViewTest(merge_enc,pxy_merge,**argsdict)
			y_bamergetwo_est = ba_mergetwo_test.test(x_mergetwo_test,y_test)
			ba_mergetwo_acc =np.sum((y_test == y_bamergetwo_est))/y_test.shape[0]
			print('BA-MergeTwo: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma,ba_mergetwo_acc *100))
			# store merge view result
			ba_merge_res[itcnt,:] = [gamma,ba_merge_out['IXZ'],ba_merge_out['IYZ'],ba_merge_out['niter'],ba_mergetwo_acc,ba_merge_out['conv']]

			# single-view baseline
			ba_v1_out = alg.ib_orig(**{'pxy':pxy_list[0],'gamma':gamma,**alg_args,**sys_param})
			v1_enc = ba_v1_out['prob_zcx']
			bav1_test = mt.singleViewTest(v1_enc,pxy_list[0],**argsdict)
			y_bav1_est = bav1_test.test(x_test_nv[:,0],y_test)
			ba_v1_acc = np.sum(np.equal(y_bav1_est,y_test))/y_test.shape[0]
			print('BA-View 1: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma,ba_v1_acc * 100))
			# store view 1 result
			ba_view1_res[itcnt,:] = [gamma,ba_v1_out['IXZ'],ba_v1_out['IYZ'],ba_v1_out['niter'],ba_v1_acc,ba_v1_out['conv']]

			ba_v2_out = alg.ib_orig(**{'pxy':pxy_list[1], 'gamma':gamma, **alg_args,**sys_param})
			v2_enc = ba_v2_out['prob_zcx']
			bav2_test = mt.singleViewTest(v2_enc,pxy_list[1],**argsdict)
			y_bav2_est = bav2_test.test(x_test_nv[:,1],y_test)
			ba_v2_acc = np.sum(np.equal(y_bav2_est,y_test))/y_test.shape[0]
			print('BA-View 2: gamma={:>6.3f}, Accuracy={:>6.3f}'.format(gamma,ba_v2_acc*100))
			# store view 2 result
			ba_view2_res[itcnt,:] = [gamma,ba_v2_out['IXZ'],ba_v2_out['IYZ'],ba_v2_out['niter'],ba_v2_acc,ba_v2_out['conv']]
			itcnt+=1
	# saving the results
	merge_template = 'bamerge_{}_acc_{}.mat'
	v1_template = 'baview1_{}_acc_{}.mat'
	v2_template = 'baview2_{}_acc_{}.mat'
	saveBaResult(merge_template,ba_merge_res,res_hdr,**argsdict)
	saveBaResult(v1_template,ba_view1_res,res_hdr,**argsdict)
	saveBaResult(v2_template,ba_view2_res,res_hdr,**argsdict)
	
elif argsdict['method'] == "inc":
	res_hdr ="gamma,penalty,IZX,IZY,IXZC,IYZC,test_acc,conv"
	#inc_res = [] # because it's possible that there is no convergence
	inc_res = np.zeros((gamma_range.shape[0]*penalty_range.shape[0]*argsdict['niter'],8))
	itcnt = 0
	for gamma in gamma_range:
		gamma_vec = np.array([gamma]*nview)
		for penc in penalty_range:
			for nn in range(argsdict['niter']):
				mv_outdict = alg.mvib_inc(**{'penalty_coefficient':penc,'gamma_vec':gamma_vec,**alg_args,**sys_param})
				if mv_outdict['conv']:
					mvinc_test = mt.mvIncTestTwo(mv_outdict['enc_list'],mv_outdict['dec_list'],pxy_list,**argsdict)
					y_inc_est = mvinc_test.test(x_test_nv,y_test)
					test_acc = np.sum(np.equal(y_test,y_inc_est))/y_test.shape[0]
					print('MV-inc: gamma={:>6.3f}, penalty={:>6.3f}, Accuracy={:>6.3f}'.format(gamma,penc,test_acc*100))
					inc_res[itcnt,:]=[gamma,penc,mv_outdict['IXZ_list'][0],mv_outdict['IYZ_list'][0],
						mv_outdict['IXZ_list'][1],mv_outdict['IYZ_list'][1],test_acc,1]
				else:
					inc_res[itcnt,:] = [gamma,penc,0.0,0.0,0.0,0.0,0.0,0]
				itcnt+=1
	inc_template = 'mvinc_{}_acc_{}.mat'
	saveBaResult(inc_template,inc_res,res_hdr,**argsdict)
elif argsdict['method'] == "cc":
	res_hdr = "gamma_v1,gamma_v2,penalty,IZX_0,IZX_1,IZY_0,IZY_1,IXZC_0,IXZC_1,IYZC_0,IYZC_1,test_acc,conv"
	cc_res = np.zeros((gamma_range.shape[0]*penalty_range.shape[0]*argsdict['niter'],13))
	itcnt =0
	#g_split_ratio = 1.0
	for gamma in gamma_range:
		# this is for the min I(X;Y), while the second range is <1  until 1-gamma...
		# For simplicity, divide equally
		gamma_v1 = gamma
		gamma_v2 = gamma
		gamma_vec = np.array([gamma_v1,gamma_v2])
		for penc in penalty_range:
			for nn in range(argsdict['niter']):
				mv_outdict = alg.mvib_cc(**{'penalty_coefficient':penc,'gamma_vec':gamma_vec,**alg_args,**sys_param})
				if mv_outdict['conv']:
					mvcc_test = mt.mvCcFullTest(mv_outdict['con_enc'],mv_outdict['cmpl_enc'],pxy_list,**argsdict)
					y_cc_est = mvcc_test.test(x_test_nv,y_test)
					test_acc = np.sum(np.equal(y_test,y_cc_est))/y_test.shape[0]
					print('MV-cc: gamma_v1={:>6.3f}, gamma_v2={:>6.3f}, Accuracy={:>6.3f}'.format(gamma_v1,gamma_v2,test_acc*100))
					mizx_list= mv_outdict['IXZ_list']
					mizy_list= mv_outdict['IYZ_list']
					mizxc_list = mv_outdict['IXZC_list']
					mizyc_list = mv_outdict['IYZC_list']
					cc_res[itcnt,:] = [gamma_v1,gamma_v2, penc, mizx_list[0], mizx_list[1], mizy_list[0],mizy_list[1],
										mizxc_list[0],mizxc_list[1],mizyc_list[0],mizyc_list[1], test_acc,1]
				else:
					cc_res[itcnt,:] = [gamma_v1,gamma_v2,penc,0,0,0,0,0,0,0,0,0,0]
				itcnt+=1
	cc_template = 'mvcc_{}_acc_{}.mat'
	saveBaResult(cc_template,cc_res,res_hdr,**argsdict)

elif argsdict['method'] == "var_2v":
	# FIXME: debugging
	gamma_vec = np.array([0.1, 0.1])
	out_dict = alg.varinf_ba2v(**{'gamma_vec':gamma_vec,**alg_args,**sys_param})
	if out_dict['conv']:
		print('varinf_2v converged')
		print("iterations:",out_dict['niter'])
		print("IXZ:",out_dict['IXZ_list'])
		print("IZCY:",out_dict['IZCY'])
	else:
		print('varinf_2v diverged')

