import numpy as np
import mvgrad as gd
import mvutils as ut
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import sys
import multiprocessing as mp
import pprint

def select(method):
	if method == "nview":
		return mvib_nv
	elif method == "parallel":
		return mvib_nv_parallel
	elif method == "reverse":
		return mvib_nv_rev
	elif method == "complement":
		return mvib_nv_cc
	else:
		sys.exit("ERROR[mvalg]: selection failed, the method {} is undefined".format(method))
def availableAlgs():
	return ['nview','parallel','reverse','complement']

# A general n view ADMM IB
def mvib_nv(pxy_list,gamma_vec,nz,convthres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# assume py are all the same for each view's joint prob
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]

	# initialization
	pzcx_list = [rs.rand(nz,i.shape[0]) for i in pxy_list]
	# normalization
	pzcx_list = [i/np.sum(i,axis=0) for i in pzcx_list]
	pz = 1.0/nview*sum([pzcx_list[i]@px_list[i] for i in range(nview)])
	pz /= np.sum(pz)
	pzcy = 1.0/nview*sum([pzcx_list[i]@pxcy_list[i] for i in range(nview)])
	pzcy /= np.sum(pzcy,axis=0)[None,:]
	# init:dual var
	dz_list = [np.zeros(nz) for i in range(nview)]
	dzcy_list = [np.zeros((nz,ny)) for i in range(nview)]

	# function and gradient objects
	fobj_pzcx_list = [gd.funcPzcxObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff) for idx in range(nview)]
	fobj_pz = gd.funcPzObj(gamma_vec,px_list,pen_coeff)
	fobj_pzcy=gd.funcPzcyObj(pxcy_list,py,pen_coeff)
	gobj_pzcx_list = [gd.gradPzcxObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff) for idx in range(nview)]
	gobj_pz = gd.gradPzObj(gamma_vec,px_list,pen_coeff)
	gobj_pzcy=gd.gradPzcyObj(pxcy_list,py,pen_coeff)

	# counter and flags
	itcnt = 0
	flag_conv = False
	while itcnt < maxiter:
		itcnt += 1
		# update primal variables
		_ss_interrupt = False
		new_pzcx_list = [np.zeros((nz,i.shape[0])) for i in pxy_list]
		for i in range(nview):
			tmp_grad_pzcx = gobj_pzcx_list[i](pzcx_list[i],pz,pzcy,dz_list[i],dzcy_list[i])
			mean_tmp_grad_pzcx = tmp_grad_pzcx - np.mean(tmp_grad_pzcx,axis=0)[None,:]
			tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[i],-mean_tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzcx==0:
				_ss_interrupt = True
				break
			'''
			tmp_ss_pzcx = gd.armijoStepSize(pzcx_list[i],-mean_tmp_grad_pzcx,tmp_ss_pzcx,gd_ss_scale,1e-4,\
								fobj_pzcx_list[i],gobj_pzcx_list[i],\
								**{'pz':pz,'pzcy':pzcy,'mu_z':dz_list[i],'mu_zcy':dzcy_list[i]})
			if tmp_ss_pzcx==0:
				_ss_interrupt = True
				break
			'''
			new_pzcx_list[i] = pzcx_list[i] -tmp_ss_pzcx * mean_tmp_grad_pzcx

		if _ss_interrupt:
			break
		# update the augmented var
		grad_pz = gobj_pz(pz,new_pzcx_list,dz_list)
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale)
		if ss_z == 0:
			break
		
		# armijo condition
		#ss_z = gd.armijoStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale,1e-4,fobj_pz,gobj_pz,\
		#			**{'pzcx_list':new_pzcx_list,'muz_list':dz_list})
		#if ss_z == 0:
		#	break
		
		grad_pzcy = gobj_pzcy(pzcy,new_pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)[None,:]
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		# armijo condition
		
		#ss_zcy = gd.armijoStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale,1e-4,fobj_pzcy,gobj_pzcy,\
		#			**{'pzcx_list':new_pzcx_list,'muzcy_list':dzcy_list})
		#if ss_zcy == 0:
		#	break
		
		# NOTE: the two augmented variables should be updated together
		ss_zzcy = min(ss_z,ss_zcy)
		new_pz = pz -ss_zzcy * mean_grad_pz
		new_pzcy = pzcy - mean_grad_pzcy * ss_zzcy
		
		# update the dual variables
		# first, calculate the errors
		errz_list  = [new_pz - np.sum(item*(px_list[idx])[None,:],axis=1) for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [new_pzcy- item@pxcy_list[idx] for idx,item in enumerate(new_pzcx_list)]
		dz_list = [ item + pen_coeff*(errz_list[idx]) for idx,item in enumerate(dz_list)]
		dzcy_list=[ item + pen_coeff*(errzcy_list[idx]) for idx,item in enumerate(dzcy_list)]
		# convergence criterion
		conv_z_list = [0.5*np.sum(np.fabs(item))<convthres for item in errz_list]
		conv_zcy_list = [0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzcy_list]
		if np.all(np.array(conv_z_list)) and np.all(np.array(conv_zcy_list)):
			flag_conv = True
			break
		# update registers
		pzcx_list = new_pzcx_list
		pz = new_pz
		pzcy = new_pzcy
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv}
def _pzcx_parallel(gamma,px,pxcy,pcoeff,pzcx_old,pz,pzcy,muz,muzcy,gd_ss_init,gd_ss_scale):
	tmp_grad = gd.gradPzcxCalc(gamma,px,pxcy,pcoeff,pzcx_old,pz,pzcy,muz,muzcy)
	mean_tmp_grad = tmp_grad - np.mean(tmp_grad,axis=0)[None,:]
	tmp_ss = gd.naiveStepSize(pzcx_old,-mean_tmp_grad,gd_ss_init,gd_ss_scale)
	new_pzcx = pzcx_old - tmp_ss*mean_tmp_grad
	div_flag = tmp_ss == 0
	return new_pzcx, div_flag
# multiprocessing for acceleration
def mvib_nv_parallel(pxy_list,gamma_vec,nz,convthres,maxiter,**kwargs):
	
	# inner function for parallel multiprocessing
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# assume py are all the same for each view's joint prob
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]

	# initialization
	pzcx_list = [rs.rand(nz,i.shape[0]) for i in pxy_list]
	# normalization
	pzcx_list = [i/np.sum(i,axis=0) for i in pzcx_list]
	pz = 1.0/nview*sum([pzcx_list[i]@px_list[i] for i in range(nview)])
	pz /= np.sum(pz)
	pzcy = 1.0/nview*sum([pzcx_list[i]@pxcy_list[i] for i in range(nview)])
	pzcy /= np.sum(pzcy,axis=0)[None,:]
	# init:dual var
	dz_list = [np.zeros(nz) for i in range(nview)]
	dzcy_list = [np.zeros((nz,ny)) for i in range(nview)]

	# function and gradient objects
	fobj_pz = gd.funcPzObj(gamma_vec,px_list,pen_coeff)
	fobj_pzcy=gd.funcPzcyObj(pxcy_list,py,pen_coeff)
	gobj_pz = gd.gradPzObj(gamma_vec,px_list,pen_coeff)
	gobj_pzcy=gd.gradPzcyObj(pxcy_list,py,pen_coeff)

	# counter and flags
	itcnt = 0
	flag_conv = False
	while itcnt < maxiter:
		itcnt += 1
		# update primal variables

		# FIXME: this part actually can be parallelly computed
		_ss_interrupt = False
		# NOTE: try mp.starmap
		para_args_list = [(gamma_vec[i],px_list[i],pxcy_list[i],pen_coeff,
							pzcx_list[i],pz,pzcy,dz_list[i],dzcy_list[i],
							gd_ss_init,gd_ss_scale) for i in range(nview)]
		new_pzcx_list = None
		with mp.Pool(processes=nview) as pool:
			new_pzcx_list, tmp_flags = zip(*pool.starmap(_pzcx_parallel,para_args_list))
			_ss_interrupt = np.any(tmp_flags)
		if _ss_interrupt:
			break
		# update the augmented var
		grad_pz = gobj_pz(pz,new_pzcx_list,dz_list)
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale)
		if ss_z == 0:
			break
		grad_pzcy = gobj_pzcy(pzcy,new_pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)[None,:]
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		# NOTE: the two augmented variables should be updated together
		ss_zzcy = min(ss_z,ss_zcy)
		new_pz = pz -ss_zzcy * mean_grad_pz
		new_pzcy = pzcy - mean_grad_pzcy * ss_zzcy
		
		# update the dual variables
		# first, calculate the errors
		errz_list  = [new_pz - np.sum(item*(px_list[idx])[None,:],axis=1) for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [new_pzcy- item@pxcy_list[idx] for idx,item in enumerate(new_pzcx_list)]
		dz_list = [ item + pen_coeff*(errz_list[idx]) for idx,item in enumerate(dz_list)]
		dzcy_list=[ item + pen_coeff*(errzcy_list[idx]) for idx,item in enumerate(dzcy_list)]
		# convergence criterion
		conv_z_list = [0.5*np.sum(np.fabs(item))<convthres for item in errz_list]
		conv_zcy_list = [0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzcy_list]
		if np.all(np.array(conv_z_list)) and np.all(np.array(conv_zcy_list)):
			flag_conv = True
			break
		# update registers
		pzcx_list = new_pzcx_list
		pz = new_pz
		pzcy = new_pzcy
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv}

# A general n view ADMM IB in reverse order
def mvib_nv_rev(pxy_list,gamma_vec,nz,convthres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# assume py are all the same for each view's joint prob
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]

	# initialization
	pzcx_list = [rs.rand(nz,i.shape[0]) for i in pxy_list]
	# normalization
	pzcx_list = [i/np.sum(i,axis=0) for i in pzcx_list]
	pz = 1.0/nview*sum([pzcx_list[i]@px_list[i] for i in range(nview)])
	pz /= np.sum(pz)
	pzcy = 1.0/nview*sum([pzcx_list[i]@pxcy_list[i] for i in range(nview)])
	pzcy /= np.sum(pzcy,axis=0)[None,:]
	# init:dual var
	dz_list = [np.zeros(nz) for i in range(nview)]
	dzcy_list = [np.zeros((nz,ny)) for i in range(nview)]

	# function and gradient objects
	fobj_pzcx_list = [gd.funcPzcxObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff) for idx in range(nview)]
	fobj_pz = gd.funcPzObj(gamma_vec,px_list,pen_coeff)
	fobj_pzcy=gd.funcPzcyObj(pxcy_list,py,pen_coeff)
	gobj_pzcx_list = [gd.gradPzcxObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff) for idx in range(nview)]
	gobj_pz = gd.gradPzObj(gamma_vec,px_list,pen_coeff)
	gobj_pzcy=gd.gradPzcyObj(pxcy_list,py,pen_coeff)

	# counter and flags
	itcnt = 0
	flag_conv = False
	while itcnt < maxiter:
		itcnt += 1
		# update primal variables

		# FIXME: this part actually can be parallelly computed
		# TODO: understand python parallelism. Multprocessing?
		_ss_interrupt = False
		# update the augmented var
		grad_pz = gobj_pz(pz,pzcx_list,dz_list)
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale)
		if ss_z == 0:
			break
		grad_pzcy = gobj_pzcy(pzcy,pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)[None,:]
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		# NOTE: the two augmented variables should be updated together
		ss_zzcy = min(ss_z,ss_zcy)
		new_pz = pz -ss_zzcy * mean_grad_pz
		new_pzcy = pzcy - mean_grad_pzcy * ss_zzcy
		# update the primal variables
		new_pzcx_list = [np.zeros((nz,i.shape[0])) for i in pxy_list]
		for i in range(nview):
			tmp_grad_pzcx = gobj_pzcx_list[i](pzcx_list[i],new_pz,new_pzcy,dz_list[i],dzcy_list[i])
			mean_tmp_grad_pzcx = tmp_grad_pzcx - np.mean(tmp_grad_pzcx,axis=0)[None,:]
			tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[i],-mean_tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzcx==0:
				_ss_interrupt = True
				break
			new_pzcx_list[i] = pzcx_list[i] -tmp_ss_pzcx * mean_tmp_grad_pzcx

		if _ss_interrupt:
			break
		
		# update the dual variables
		# first, calculate the errors
		errz_list  = [new_pz - np.sum(item*(px_list[idx])[None,:],axis=1) for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [new_pzcy- item@pxcy_list[idx] for idx,item in enumerate(new_pzcx_list)]
		dz_list = [ item + pen_coeff*(errz_list[idx]) for idx,item in enumerate(dz_list)]
		dzcy_list=[ item + pen_coeff*(errzcy_list[idx]) for idx,item in enumerate(dzcy_list)]
		# convergence criterion
		conv_z_list = [0.5*np.sum(np.fabs(item))<convthres for item in errz_list]
		conv_zcy_list = [0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzcy_list]
		if np.all(np.array(conv_z_list)) and np.all(np.array(conv_zcy_list)):
			flag_conv = True
			break
		# update registers
		pzcx_list = new_pzcx_list
		pz = new_pz
		pzcy = new_pzcy
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv}

# the new formulation, with complement information taken into consideration
def mvib_nv_cc(pxy_list,gamma_vec,gamma_cmpl,nzc,nze_vec,convthres,maxiter,**kwargs):
	# besides the dim for common representation zc,
	# the dimensions for the complement representations zn for each view should be given
	# This will result in additional sets of augmented variables and even coupling of them.
	# should carefully handle the update steps!
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# assume py are all the same for each view's joint prob
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]
	pycx_list = [(item/px_list[idx][:,None]).T for idx,item in enumerate(pxy_list)]
	# initialization
	# NOTE: this is kept as a tensor
	# p complement, joint probability
	pzeccx_list = [rs.rand(nze_vec[idx],nzc,item.shape[0]) for idx,item in enumerate(pxy_list)] # the dimension as in the x view
	# how to normalize a tensor?
	pzeccx_list = [tt/np.sum(tt,axis=(0,1))[...,:] for tt in pzeccx_list]
	# initialize with summing the random point in complement view
	# p common
	pzcx_list = [np.sum(tt,axis=0) for tt in pzeccx_list]
	# q common
	pz_cmon = 1/nview * sum([pzcx_list[i]@px_list[i] for i in range(nview)])
	pz_cmon /= np.sum(pz_cmon)
	pzcy_cmon = 1/nview * sum([pzcx_list[i]@pxcy_list[i] for i in range(nview)])
	pzcy_cmon /= np.sum(pzcy_cmon,axis=0)[None,:]
	# dual variable
	dz_list = [np.zeros(nzc) for i in range(nview)]
	dzcy_list = [np.zeros((nzc,ny)) for i in range(nview)]

	dzec_list = [np.zeros((nzc,pxy_list[idx].shape[0])) for idx in range(nview)]
	# gradient objects, possibly function value objects
	#gobj_pzcx_list = [gd.gradPzcxComnObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff) for idx in range(nview)]
	# CAUTION: masked gradient subtract mean within the object. don't need to do this step afterward
	gobj_pzcx_list = [gd.maskGradPzcxCmonObj(gamma_vec[idx],px_list[idx],pxcy_list[idx],pen_coeff,1e-10) for idx in range(nview)]
	gobj_pz = gd.gradPzComnObj(gamma_vec,px_list,pen_coeff)
	gobj_pzcy=gd.gradPzcyComnObj(pxcy_list,py,pen_coeff)
	#gobj_pzeccx_list = [gd.gradPzcxCmplObj(gamma_cmpl,px_list[idx],pxcy_list[idx],pycx_list[idx],pen_coeff) for idx in range(nview)]
	gobj_pzeccx_list = [gd.maskGradPzcxCmplObj(gamma_cmpl,px_list[idx],pxcy_list[idx],pycx_list[idx],pen_coeff,1e-10) for idx in range(nview)]

	# counters and registers
	itcnt =0
	flag_conv = False
	while itcnt< maxiter:
		itcnt += 1
		# step 1: update view common encoder
		_ss_interrupt = False
		new_pzcx_list = [np.zeros((nzc,tt.shape[0])) for tt in pxy_list]
		for vi in range(nview):
			tmp_grad_pzcx = gobj_pzcx_list[vi](pzcx_list[vi],pz_cmon,pzcy_cmon,pzeccx_list[vi],dz_list[vi],dzcy_list[vi],dzec_list[vi])
			#mean_tmp_grad_pzcx = tmp_grad_pzcx - np.mean(tmp_grad_pzcx,axis=0)[None,:]
			#tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[vi],-mean_tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[vi],-tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzcx == 0:
				_ss_interrupt = True
				break
			#new_pzcx_list[vi] = pzcx_list[vi] - tmp_ss_pzcx*mean_tmp_grad_pzcx
			new_pzcx_list[vi] = pzcx_list[vi] - tmp_ss_pzcx*tmp_grad_pzcx
		if _ss_interrupt:
			#print('DEBUG: INTERRUPT, COMMON PZCX')
			break
		# step 2: update the common latent representation
		grad_pz_cmon = gobj_pz(pz_cmon,new_pzcx_list,dz_list)
		mean_grad_pz_cmon = grad_pz_cmon - np.mean(grad_pz_cmon)
		ss_z_cmon = gd.naiveStepSize(pz_cmon,-mean_grad_pz_cmon,gd_ss_init,gd_ss_scale)
		if ss_z_cmon == 0:
			#print('DEBUG: INTERRUPT, COMMON PZ')
			break
		grad_pzcy_cmon = gobj_pzcy(pzcy_cmon,new_pzcx_list,dzcy_list)
		mean_grad_pzcy_cmon = grad_pzcy_cmon - np.mean(grad_pzcy_cmon,axis=0)[None,:]
		ss_zcy_cmon = gd.naiveStepSize(pzcy_cmon,-mean_grad_pzcy_cmon,gd_ss_init,gd_ss_scale)
		if ss_zcy_cmon ==0:
			#print('DEBUG: INTERRUPT, COMMON PZCY')
			break
		ss_zzcy_cmon_min = min(ss_z_cmon,ss_zcy_cmon) #NOTE: the update should be coupled together
		new_pz_cmon = pz_cmon - mean_grad_pz_cmon * ss_zzcy_cmon_min
		new_pzcy_cmon = pzcy_cmon - mean_grad_pzcy_cmon * ss_zzcy_cmon_min
		# step 3: update view complement encoder
		# NOTE: Can be done once pzcx is updated. Written here for development.
		#       Each complement encoder is stored as a tensor and is a joint probability
		new_pzeccx_list = [np.zeros((nze_vec[idx],nzc,tt.shape[0])) for idx,tt in enumerate(pxy_list)]
		for vi in range(nview):
			tmp_grad_pzeccx = gobj_pzeccx_list[vi](pzeccx_list[vi],new_pzcx_list[vi],dzec_list[vi])
			#mean_tmp_grad_pzeccx = tmp_grad_pzeccx - np.mean(tmp_grad_pzeccx,axis=(0,1))[...,:]
			#tmp_ss_pzeccx = gd.naiveStepSize(pzeccx_list[vi],-mean_tmp_grad_pzeccx,gd_ss_init,gd_ss_scale)
			# CAUTION: if using the masked gradient, then no mean subtraction is needed
			tmp_ss_pzeccx = gd.naiveStepSize(pzeccx_list[vi],-tmp_grad_pzeccx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzeccx == 0:
				_ss_interrupt = True
				print('DEBUG: INTERRUPT, COMPLEMENT PZECCX')
				pprint.pprint({'var':pzeccx_list[vi],'grad':tmp_grad_pzeccx,'grad_sum':np.sum(tmp_grad_pzeccx,axis=(0,1))})
				print('DEBUG END')
				break
			#new_pzeccx_list[vi] = pzeccx_list[vi] - tmp_ss_pzeccx * mean_tmp_grad_pzeccx
			new_pzeccx_list[vi] = pzeccx_list[vi] - tmp_ss_pzeccx * tmp_grad_pzeccx
		if _ss_interrupt:
			break
		# step 4: dual variables updates
		# common error
		errz_list = [ np.sum(item*px_list[idx][None,:],axis=1)-new_pz_cmon for idx,item in enumerate(new_pzcx_list)]
		errzcy_list = [item@pxcy_list[idx]-new_pzcy_cmon for idx, item in enumerate(new_pzcx_list)]
		errzec_list = [new_pzcx_list[idx]-np.sum(item,axis=0) for idx, item in enumerate(new_pzeccx_list)]
		dz_list = [item + pen_coeff * (errz_list[idx]) for idx,item in enumerate(dz_list)]
		dzcy_list = [item + pen_coeff * (errzcy_list[idx]) for idx,item in enumerate(dzcy_list)]
		dzec_list = [item + pen_coeff * (errzec_list[idx]) for idx,item in enumerate(dzec_list)]
		#print('-'*50)
		#print(dzec_list)
		# complement error
		# Control step: convergence criterion
		conv_z_list = np.array([0.5* np.sum(np.fabs(item))<convthres for item in errz_list])
		conv_zcy_list = np.array([0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzcy_list])
		#conv_zec_list = np.array([True])
		conv_zec_list = np.array([np.all(tt) for tt in [0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzec_list]])
		conv_all = np.all(conv_zcy_list) and np.all(conv_z_list) and np.all(conv_zec_list)
		if conv_all:
			flag_conv = True
			break
		# Control step: passing to next iteration
		pzcx_list = new_pzcx_list
		pz_cmon = new_pz_cmon
		pzcy_cmon = new_pzcy_cmon
		pzeccx_list = new_pzeccx_list
	errz_list = [np.sum(item*px_list[idx][None,:],axis=1)-pz_cmon for idx,item in enumerate(pzcx_list)]
	errzcy_list = [item@pxcy_list[idx]-pzcy_cmon for idx, item in enumerate(pzcx_list)]
	errzec_list = [pzcx_list[idx]-np.sum(item,axis=0) for idx, item in enumerate(pzeccx_list)]
	print('-'*50)
	#print(errz_list)
	#print(errzcy_list)
	#print(errzec_list)
	#print(pzcx_list)
	#print(pzeccx_list)
	#print([np.sum(item,axis=0) for item in pzeccx_list])
	return {'pzcx_list':pzcx_list,'pz':pz_cmon,'pzcy':pzcy_cmon,
			'pzcx_cmpl_list':pzeccx_list,
			'niter':itcnt,'conv':flag_conv}

# compared algorithms
'''
def ib_orig(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pycx = np.transpose(np.diag(1./px)@pxy)
	pxcy = pxy@np.diag(1./py)
	# on IB, the initialization matters
	# use random start (*This is the one used for v2)
	sel_idx = np.random.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	
	# use random start
	#sel_idx = np.random.permutation(nx)
	#pycz = pycx[:,sel_idx[:qlevel]]
	#pz = px[sel_idx[:qlevel]]
	#pz /= np.sum(pz)
	#pzcx = np.ones((nz,nx))/nz
	
	# ready to start
	itcnt = 0
	while itcnt<max_iter:
		# compute ib kernel
		new_pzcx= np.zeros((nz,nx))
		kl_oprod = np.expand_dims(1./pycz,axis=-1)@np.expand_dims(pycx,axis=1)
		kl_ker = np.sum(np.repeat(np.expand_dims(pycx,axis=1),nz,axis=1)*np.log(kl_oprod),axis=0)
		new_pzcx = np.diag(pz)@np.exp(-beta*kl_ker)

		# standard way, normalize to be valid probability.
		new_pzcx = new_pzcx@np.diag(1./np.sum(new_pzcx,axis=0))
		itcnt+=1
		# total variation convergence criterion
		diff = 0.5* np.sum(np.fabs(new_pzcx-pzcx))
		if diff < conv_thres:
			# reaching convergence
			break
		else:
			# update other probabilities
			pzcx = new_pzcx
			# NOTE: should use last step? or the updated one?
			pz = pzcx@px
			pzcy = pzcx@pxcy
			pycz = np.transpose(np.diag(1./pz)@pzcy@np.diag(py))
	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pycz,pz)
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':True}
'''

