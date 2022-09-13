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
	if method == "cc":
		return mvib_cc
	#if method == "nview":
	#	return mvib_nv
	#elif method == "parallel":
	#	return mvib_nv_parallel
	#elif method == "nvavg":
	#	return mvib_nv_avg
	elif method == "inc":
		return mvib_inc
	elif method == "ba":
		return ib_orig
	elif method == "var_2v":
		return varinf_ba2v
	else:
		sys.exit("ERROR[mvalg]: selection failed, the method {} is undefined".format(method))
def availableAlgs():
	#return ['nview','parallel','mvavg','increment']
	return ["cc","inc",'ba','var_2v']

# A general n view ADMM IB
def mvib_nv(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
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
	nz = np.amin([item.shape[0] for item in pxy_list]) # shrink the representation dimension to the min nx
	# initialization
	pzcx_list = [rs.rand(nz,i.shape[0]) for i in pxy_list]
	# deterministic start
	'''
	d_smooth_factor = 1e-2
	pzcx_list = []
	for idx  in range(nview):
		# assume nz <= min(nx) (for all view)
		init_mat = np.zeros((nz,px_list[idx].shape[0]))
		init_mat[:,0:nz] = np.eye(nz,dtype=float)
		# random permutation
		if nz < px_list[idx].shape[0]:
			partial_perm = rs.randint(nz,size=px_list[idx].shape[0]-nz)
			for iner in range(len(partial_perm)):
				init_mat[partial_perm[iner],nz+iner] = 1.0
		tmp_perm = rs.permutation(px_list[idx].shape[0])
		raw_mat = init_mat[:,tmp_perm] + d_smooth_factor
		raw_mat /= np.sum(raw_mat,axis=0)
		pzcx_list.append(raw_mat)
	'''
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

	# debugging
	#min_mass_pzcy = np.zeros((maxiter))

	# counter and flags
	itcnt = 0
	flag_conv = False
	while itcnt < maxiter:
		#min_mass_pzcy[itcnt] = np.amin(pzcy)
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
			new_pzcx_list[i] = pzcx_list[i] -tmp_ss_pzcx * mean_tmp_grad_pzcx

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
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,ss_z,gd_ss_scale)
		if ss_zcy == 0:
			break
		
		# NOTE: the two augmented variables should be updated together
		new_pz = pz -ss_zcy * mean_grad_pz
		new_pzcy = pzcy - mean_grad_pzcy * ss_zcy
		
		# update the dual variables
		# first, calculate the errors
		errz_list  = [np.sum(item*(px_list[idx])[None,:],axis=1)-new_pz for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [item@pxcy_list[idx]-new_pzcy for idx,item in enumerate(new_pzcx_list)]
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
	mizx_list = [ut.calcMI(pzcx_list[i]*px_list[i][None,:]) for i in range(nview)]
	mizy_list = [ut.calcMI(pzcx_list[i]@pxy_list[i]) for i in range(nview)]
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv,'IXZ_list':mizx_list,'IYZ_list':mizy_list}

def mvib_nv_avg(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
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
	#nz = np.amax([item.shape[0] for item in pxy_list]) # expand the representation dimension to the max nx
	nz = np.amin([item.shape[0] for item in pxy_list]) # shrink the representation dimension to the min nx
	# initialization
	# FIXME: could set to deterministic method
	pzcx_list = [rs.rand(nz,i.shape[0]) for i in pxy_list]
	# deterministic start
	'''
	d_smooth_factor = 1e-3
	pzcx_list = []
	for idx  in range(nview):
		# assume nz <= min(nx) (for all view)
		init_mat = np.zeros((nz,px_list[idx].shape[0]))
		init_mat[:,0:nz] = np.eye(nz,dtype=float)
		# random permutation
		if nz < px_list[idx].shape[0]:
			partial_perm = rs.randint(nz,size=px_list[idx].shape[0]-nz)
			for iner in range(len(partial_perm)):
				init_mat[partial_perm[iner],nz+iner] = 1.0
		tmp_perm = rs.permutation(px_list[idx].shape[0])
		raw_mat = init_mat[:,tmp_perm] + d_smooth_factor
		raw_mat /= np.sum(raw_mat,axis=0)
		pzcx_list.append(raw_mat)
	'''
	# normalization
	pzcx_list = [i/np.sum(i,axis=0) for i in pzcx_list]
	pz = 1.0/nview*sum([pzcx_list[i]@px_list[i] for i in range(nview)])
	pz /= np.sum(pz)
	pzcy = 1.0/nview*sum([pzcx_list[i]@pxcy_list[i] for i in range(nview)])
	pzcy /= np.sum(pzcy,axis=0)

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
			new_pzcx_list[i] = pzcx_list[i] -tmp_ss_pzcx * mean_tmp_grad_pzcx

		if _ss_interrupt:
			break
		# update the augmented var
		grad_pzcy = gobj_pzcy(pzcy,new_pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)[None,:]
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		grad_pz = gobj_pz(pz,new_pzcx_list,dz_list)
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,ss_zcy,gd_ss_scale)
		if ss_z == 0:
			break
		
		# NOTE: the two augmented variables should be updated together
		new_pz = pz -ss_z * mean_grad_pz
		new_pzcy = pzcy - ss_z * mean_grad_pzcy
		# update the dual variables
		# first, calculate the errors
		errz_list  = [np.sum(item* (px_list[idx][None,:]),axis=1)-new_pz for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [item@pxcy_list[idx]-new_pzcy for idx,item in enumerate(new_pzcx_list)]
		dz_list = [ item + pen_coeff*errz_list[idx] for idx,item in enumerate(dz_list)]
		dzcy_list=[ item + pen_coeff*errzcy_list[idx] for idx,item in enumerate(dzcy_list)]
		# averaging the dual
		avg_dz= sum(dz_list)/nview
		avg_dzy=sum(dzcy_list)/nview
		dz_list = [avg_dz]*nview
		dzcy_list = [avg_dzy]*nview

		# convergence criterion
		#conv_z_list = [np.linalg.norm(item)<convthres for item in errz_list]
		#conv_zcy_list = [np.linalg.norm(item,axis=0)<convthres for item in errzcy_list]
		conv_z_list = [0.5*np.sum(np.fabs(item))<convthres for item in errz_list]
		conv_zcy_list = [0.5*np.sum(np.fabs(item),axis=0)<convthres for item in errzcy_list]
		if np.all(np.array(conv_z_list)) and np.all(np.array(conv_zcy_list)):
			flag_conv = True
			break
		else:
			# update registers
			pzcx_list = new_pzcx_list
			pz = new_pz
			pzcy = new_pzcy	
	# post calculation
	# DEBUG
	
	#errz_list =  [np.sum(item*(px_list[idx])[None,:],axis=1)-pz for idx,item in enumerate(pzcx_list)]
	#errzy_list = [item@pxcy_list[idx]-pzcy for idx,item in enumerate(pzcx_list)]
	#conv_z_list = [0.5*np.sum(np.fabs(item)) for item in errz_list]
	#conv_zcy_list = [0.5*np.sum(np.fabs(item),axis=0) for item in errzcy_list]
	mizx_list = [ut.calcMI(pzcx_list[i]*px_list[i][None,:]) for i in range(nview)]
	mizy_list = [ut.calcMI(pzcx_list[i]@pxy_list[i]) for i in range(nview)]
	#print('final residual:',conv_z_list,conv_zcy_list)
	#print('MIXY',[ut.calcMI(item) for item in pxy_list])
	#print('MIXZ',mizx_list)
	#print('MIYZ',mizy_list)
	
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv,'IXZ_list':mizx_list,'IYZ_list':mizy_list}

'''
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
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,ss_z,gd_ss_scale)
		if ss_zcy == 0:
			break
		# NOTE: the two augmented variables should be updated together
		new_pz = pz -ss_zcy * mean_grad_pz
		new_pzcy = pzcy - mean_grad_pzcy * ss_zcy
		
		# update the dual variables
		# first, calculate the errors
		errz_list  = [np.sum(item*(px_list[idx])[None,:],axis=1)-new_pz for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [item@pxcy_list[idx]-new_pzcy for idx,item in enumerate(new_pzcx_list)]
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
'''

# two-step method, complement representation algorithm
# simple gradient descent, not a penalty method
'''
def mvib_sv_cmpl(pxy,ne,enc_pzcx,gamma,conv_thres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# FIXME: fixed threshold
	vmask_thres = 1e-9
	#pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# preparing the prerequsite
	(nx,ny) = pxy.shape
	nc = enc_pzcx.shape[0]
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy/py[None,:]
	pycx = (pxy/px[:,None]).T
	expd_enc = np.repeat(enc_pzcx[None,:],ne,axis=0)
	# initialization
	pzeccx = rs.rand(ne,nc,nx)
	nsum = np.sum(pzeccx,axis=0)
	pzeccx /= nsum[None,:]
	pzeccx *= enc_pzcx[None,:]
	itcnt =0 
	conv_flag = False
	while itcnt< maxiter:
		itcnt += 1
		vmask = np.logical_and(pzeccx > vmask_thres , pzeccx < 1-vmask_thres)
		pzec = np.sum(pzeccx*px[None,:],axis=2)
		rep_pzec = np.repeat(pzec[:,None],nx,axis=2)
		pzeccy = pzeccx @ pxcy
		grad_yterm = np.log(pzeccy) @ pycx
		raw_grad = np.zeros((ne,nc,nx))
		raw_grad[vmask] = gamma*(np.log(pzeccx[vmask]/rep_pzec[vmask])\
						-1/gamma*(grad_yterm[vmask]-np.log(rep_pzec[vmask])))
		raw_grad *= px[None,:]
		# conditional mean
		mean_grad = np.zeros((ne,nc,nx))
		calc_grad = np.mean(raw_grad,axis=0)
		mean_grad[vmask] = raw_grad[vmask]- np.repeat(calc_grad[None,:],ne,axis=0)[vmask]
		ss_tmp = gd.naiveConditionalStepSize(pzeccx,-mean_grad,expd_enc,gd_ss_init,gd_ss_scale)
		if ss_tmp ==0:
			break
		new_pzeccx = pzeccx -ss_tmp* mean_grad
		#dtv= 0.5*np.sum(np.abs(new_pzeccx-pzeccx),axis=(0,1))
		gnorm = np.linalg.norm(mean_grad,axis=(0,1))**2
		#debug_error = gnorm
		# FIXME: seems like gradient norm criterion gives better performance
		#if np.all(np.array(dtv<conv_thres)):
		if np.all(np.array(gnorm<conv_thres)):
			conv_flag = True
			break
		else:
			pzeccx = new_pzeccx
	return {'pzeccx':pzeccx,'niter':itcnt,'conv':conv_flag}
'''


def mvib_sv_cmpl_type1(pxy,enc_pzcx,gamma,convthres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# FIXME: fixed threshold
	penc = kwargs['penalty_coefficient']
	ssinit = kwargs['init_stepsize']
	sscale = kwargs['stepsize_scale']
	# preparing the prerequsite
	(nx,ny) = pxy.shape
	nc = enc_pzcx.shape[0]
	px = np.sum(pxy,axis=1)
	ne = px.shape[0]
	py = np.sum(pxy,axis=0)
	pxcy = pxy/py[None,:]
	pycx = (pxy/px[:,None]).T
	# precomputed constants
	const_pzc_x = enc_pzcx *px[None,:] # NOTE: this part still depends on x's dimension...
	const_pzc   = np.sum(enc_pzcx *px[None,:],axis=-1)
	const_pzc_y = enc_pzcx @ pxy
	prior_pzcy =enc_pzcx @ pxcy

	const_errz_x = np.repeat((const_pzc_x / const_pzc[:,None])[None,...],ne,axis=0)

	# initialization
	# FIXME: start with random initialization, could change to deterministic start
	var_pzcx = rs.rand(ne,nc,nx)
	var_pzcx /= np.sum(var_pzcx,axis=0)

	var_pz = np.sum(var_pzcx * const_pzc_x[None,...],axis=-1) / const_pzc[None,:]
	# dual variables
	dual_z = np.zeros((ne,nc))

	itcnt =0 
	conv_flag = False
	while itcnt< maxiter:
		itcnt +=1
		errz = var_pz - np.sum(var_pzcx * const_pzc_x[None,...],axis=-1)/const_pzc[None,:]
		# grad_z
		# (gamma-1) H(Ze|Zc)
		grad_z = (1-gamma)* (np.log(var_pz)+1)*const_pzc[None,:] + (dual_z + penc*errz)
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		ss_z = gd.naiveStepSize(var_pz,-mean_grad_z,ssinit,sscale)
		if ss_z == 0:
			break
		new_var_pz = var_pz - ss_z * mean_grad_z
		errz = new_var_pz - np.sum(var_pzcx * const_pzc_x[None,...],axis=-1)/const_pzc[None,:]
		# dual ascend
		# accumulate the error here
		dual_z += penc*errz
		# grad_x
		# -gamma H(Ze|Zc,X) + H(Ze|Zc,Y)
		# FIXME: this could be simplified to
		
		#tmp_pzcy = ((var_pzcx * enc_pzcx[None,...])@pxcy)/(prior_pzcy[None,...])
		#grad_x = gamma * (np.log(var_pzcx)+1) * const_pzc_x[None,:] \
		#		- ( ((np.log(tmp_pzcy)+1)/prior_pzcy[None,...]) @pxcy.T)*enc_pzcx[None,...]\
		#		-(dual_z + penc * errz)[...,None] * const_errz_x
		
		tmp_pzcy = var_pzcx @ pxcy
		grad_x = gamma * (np.log(var_pzcx)+1) * const_pzc_x[None,:] \
				- (np.log(tmp_pzcy)+1)@pxcy.T\
				-((dual_z + penc * errz)/const_pzc[None,:])[...,None]*const_pzc_x[None,...]
		
		mean_grad_x = grad_x - np.mean(grad_x,axis=0)
		ss_x= gd.naiveStepSize(var_pzcx,-mean_grad_x,ssinit,sscale)
		if ss_x ==0:
			break
		new_var_pzcx = var_pzcx - ss_x*mean_grad_x
		# error estimate
		errz = new_var_pz - np.sum(new_var_pzcx * const_pzc_x[None,...],axis=-1)/const_pzc[None,:]

		dtvz = 0.5*np.sum(np.fabs(errz),axis=0)
		if np.all(np.array(dtvz<convthres)):
			conv_flag=True
			break
		else:
			var_pzcx = new_var_pzcx
			var_pz = new_var_pz
	errz = var_pz - np.sum(var_pzcx * const_pzc_x[None,...],axis=-1)/const_pzc[None,:]
	dtvz = 0.5*np.sum(np.fabs(errz),axis=0)
	# FIXME: debugging 
	#print('residual:',errz)
	#print('dtv:',dtvz)
	joint_pzcx = var_pzcx * const_pzc_x[None,...]
	joint_pzcy = (var_pzcx@ pxcy) * const_pzc_y[None,...]
	tmp_var_pzcy = var_pzcx@pxcy
	miczx = np.sum(joint_pzcx*np.log(var_pzcx/var_pz[...,None]))
	miczy = np.sum(joint_pzcy*np.log(tmp_var_pzcy/var_pz[...,None]))
	
	return {'pzeccx':var_pzcx,'niter':itcnt,'conv':conv_flag,'IZCX':miczx,'IZCY':miczy}

	#return {'pzeccx':var_pzcx,'niter':itcnt,'conv':conv_flag}


def mvib_sv_cmpl_type2(pxy,enc_pzcx,gamma,convthres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# FIXME: fixed threshold
	penc = kwargs['penalty_coefficient']
	ssinit = kwargs['init_stepsize']
	sscale = kwargs['stepsize_scale']
	# preparing the prerequsite
	(nx,ny) = pxy.shape
	nc = enc_pzcx.shape[0]
	px = np.sum(pxy,axis=1)
	ne = px.shape[0]
	py = np.sum(pxy,axis=0)
	pxcy = pxy/py[None,:]
	pycx = (pxy/px[:,None]).T
	# some constants
	const_pzx = enc_pzcx * px[None,:]
	const_pz  = enc_pzcx@px
	const_pzy = enc_pzcx @ pxy
	const_pzcy = enc_pzcx @ pxcy
	#const_grad_x= np.repeat(const_pzx[None,...],ne,axis=0)
	const_grad_x = const_pzx/const_pz[:,None]

	var_pzcx = rs.rand(ne,nc,nx)
	var_pzcx /= np.sum(var_pzcx,axis=0)

	var_pz = np.sum(var_pzcx * enc_pzcx[None,...] * px[...,:],axis=-1) / const_pz[None,:]
	var_pzcy = ((var_pzcx * enc_pzcx[None,...])@pxcy)/((enc_pzcx@pxcy)[None,...])
	#var_pzcy = var_pzcx @ pxcy
	# dual variables
	dual_z = np.zeros((ne,nc))
	dual_zy= np.zeros((ne,nc,ny))


	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt+=1
		# type II
		errz = np.sum(var_pzcx*const_pzx[None,...],axis=-1)/(const_pz[None,:]) - var_pz
		errzy = ((var_pzcx * enc_pzcx[None,...])@pxcy)/((const_pzcy)[None,...]) - var_pzcy
		#errzy = var_pzcx @ pxcy - var_pzcy
		# -gamma_i H(Z_e|Z_c,X^i)
		grad_x = gamma*(np.log(var_pzcx)+1)*const_pzx[None,...] +(dual_z + penc*errz)[...,None]*const_grad_x[None,...]\
				+((dual_zy+penc*errzy)/const_pzcy[None,...])@pxcy.T * enc_pzcx[None,...]
		#grad_x = gamma*(np.log(var_pzcx)+1)*const_pzx[None,...] +(dual_z + penc*errz)[...,None] * const_grad_x[None,...]\
		#		+(dual_zy+penc*errzy)@pxcy.T
		mean_grad_x = grad_x - np.mean(grad_x,axis=0)
		ss_x = gd.naiveStepSize(var_pzcx,-mean_grad_x,ssinit,sscale)
		if ss_x == 0:
			break
		new_var_pzcx = var_pzcx -mean_grad_x * ss_x
		# (gamma-1) H(Z_e|Z_c)
		errz = np.sum(new_var_pzcx*const_pzx[None,...],axis=-1)/(const_pz[None,:]) - var_pz
		errzy = ((new_var_pzcx * enc_pzcx[None,...])@pxcy)/((const_pzcy)[None,...]) - var_pzcy
		#errzy = new_var_pzcx @pxcy - var_pzcy
		grad_z = (1-gamma) *(np.log(var_pz)+1)*const_pz[None,:] -(dual_z+penc*errz)
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		ss_z = gd.naiveStepSize(var_pz,-mean_grad_z,ssinit,sscale)
		if ss_z == 0:
			break
		# H(Z_e|Z_c,Y)
		grad_y = -(np.log(var_pzcy))*const_pzy[None,...] - (dual_zy+penc*errzy)
		mean_grad_y = grad_y - np.mean(grad_y,axis=0)
		ss_y = gd.naiveStepSize(var_pzcy,-mean_grad_y,ss_z,sscale)
		if ss_y == 0:
			break
		new_var_pz = var_pz - ss_y*mean_grad_z
		new_var_pzcy = var_pzcy - ss_y * mean_grad_y
		errz = np.sum(new_var_pzcx*const_pzx[None,...],axis=-1)/(const_pz[None,:]) - new_var_pz
		errzy = ((new_var_pzcx * enc_pzcx[None,...])@pxcy)/((const_pzcy)[None,...]) - new_var_pzcy
		#errzy = new_var_pzcx @ pxcy - new_var_pzcy
		# dual update
		dual_z += penc * errz
		dual_zy += penc*errzy
		dtvz = 0.5 * np.sum(np.abs(errz),axis=0)
		dtvzy= 0.5 * np.sum(np.abs(errzy),axis=0)
		if np.all(np.array(dtvz < convthres)) and np.all(np.array(dtvzy<convthres)):
			conv_flag = True
			break
		else:
			var_pzcx = new_var_pzcx
			var_pz = new_var_pz
			var_pzcy = new_var_pzcy
	joint_pzcx = var_pzcx * const_pzx[None,...]
	joint_pzcy = var_pzcy * const_pzy[None,...]
	miczx = np.sum(joint_pzcx*np.log(var_pzcx/var_pz[...,None]))
	miczy = np.sum(joint_pzcy*np.log(var_pzcy/var_pz[...,None]))
	
	return {'pzeccx':var_pzcx,'niter':itcnt,'conv':conv_flag,'IZCX':miczx,'IZCY':miczy}

def mvib_cc(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	d_retry = kwargs['retry']
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]
	# step 0: preparation, sort the MI 
	#mi_idx_sortlist = np.flip(np.argsort([ut.calcMI(i) for i in pxy_list])) # from greatest to smallest
	# step 1: run the consensus step
	#outdict = mvib_nv_avg(pxy_list,gamma_vec,convthres,maxiter,**kwargs)
	#output:{'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv,'IXZ_list':mizx_list,'IYZ_list':mizy_list}
	outdict = mvib_nv(pxy_list,gamma_vec,convthres,maxiter,**kwargs)
	if not outdict['conv']:
		# the algorithm diverged, we can either increase the penalty coefficient until convergence is assured
		print('ERROR:consensus failed')
		return {'conv':False}
	# FIXME: debugging purpose
	debug_est_pzcx = outdict['pzcx_list']
	debug_est_mizx = [ut.calcMI(item*px_list[idx][None,:]) for idx,item in enumerate(debug_est_pzcx)]
	debug_est_mizy = [ut.calcMI(item@pxy_list[idx]) for idx,item in enumerate(debug_est_pzcx)]
	print('DEBUG: MIZX=',debug_est_mizx)
	print('DEBUG: MIZY=',debug_est_mizy)
	print('LOG:consensus algorithm converged in {:>5} iterations:'.format(outdict['niter']))
	#print('MIZX:',outdict['IXZ_list'])
	#print('MIZY:',outdict['IYZ_list'])
	# step 2: run the complement step for the rest, in MI descending order
	tmp_cmpl_list = []
	cmpl_izcx_list = []
	cmpl_izcy_list = []
	for midx in range(nview):
		# FIXME
		inner_loop_conv =False
		best_mic = -1.0
		best_out = {}
		for rn in range(d_retry):
			#cmpl_out = mvib_sv_cmpl_type1(pxy_list[midx],outdict['pzcx_list'][midx],gamma_vec[midx],convthres,maxiter,**kwargs)
			cmpl_out = mvib_sv_cmpl_type2(pxy_list[midx],outdict['pzcx_list'][midx],gamma_vec[midx],convthres,maxiter,**kwargs)
			if not cmpl_out['conv']:
				#error_flag = True
				print('ERROR: view {:>5} failed (retry count:{:>3})'.format(midx,rn))
				#return {'conv':False}
			else:
				if cmpl_out['IZCY']>best_mic:
					best_mic = cmpl_out['IZCY']
					best_out = copy.deepcopy(cmpl_out)
				inner_loop_conv=True
				print('LOG: view {:>5} converged (retry count:{:>3}): best_IYZC={:>10.5f}'.format(midx,rn,best_mic))
		if not inner_loop_conv:
			return {'conv':False}
		else:
			# FIXME: for debugging
			print('LOG:complement view {:>5} converged: IXZC={:>10.4f}, IYZC={:>10.4f}'.format(midx,best_out['IZCX'],best_out['IZCY']))
			tmp_cmpl_list.append(best_out['pzeccx'])
			cmpl_izcx_list.append(best_out['IZCX'])
			cmpl_izcy_list.append(best_out['IZCY'])
	# put the complement encoders back to the order as in pxy_list
	'''
	est_list  =[None] * nview
	cmpl_mizx =[0.0] * nview
	cmpl_mizy =[0.0] * nview
	for idx in range(nview):
		est_list[mi_idx_sortlist[idx]] = tmp_cmpl_list[idx]
		cmpl_mizx[mi_idx_sortlist[idx]] = cmpl_izcx_list[idx]
		cmpl_mizy[mi_idx_sortlist[idx]] = cmpl_izcy_list[idx]
	'''
	# FIXME: this log is for debugging purpose
	print('LOG:convergence of mvib_cc reached!')
	return {'con_enc':outdict['pzcx_list'],'cmpl_enc':tmp_cmpl_list,
			'IXZ_list':outdict['IXZ_list'],'IYZ_list':outdict['IYZ_list'],
			'IXZC_list':cmpl_izcx_list,'IYZC_list':cmpl_izcy_list,'conv':True}
def mvib_inc(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	# NOTE: should think about the best value of nz
	# assume the pxy is already sorted in descending order MI
	d_retry = kwargs['retry']
	# the tensor is expanding... how to handle it efficiently?
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# assume py are all the same for each view's joint prob
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	ny = len(py)
	# generate lists
	px_list = [np.sum(i,axis=1) for i in pxy_list]
	pxcy_list = [i/py[None,:] for i in pxy_list]
	# step 0: preparation, sort the MI 
	mi_idx_sortlist = np.flip(np.argsort([ut.calcMI(i) for i in pxy_list])) # from greatest to smallest
	pzcy_prior = None
	tmp_est_list =[]
	tmp_dec_list =[]
	mizx_list = []
	mizy_list = []
	for itcnt , midx in enumerate(mi_idx_sortlist):
		if itcnt == 0:
			# step 1: run a single BA
			# the BA need "beta" instead of "gamma"
			est_init = ib_orig(pxy_list[midx],convthres,gamma_vec[midx],maxiter,**kwargs) # must converge			
			# must converge
			if est_init['conv']:
				print('LOG:incremental view {:>5} converged (BA)--IXZ={:>10.4f}, IYZ={:>10.4f}'.format(midx,est_init['IXZ'],est_init['IYZ']))
				pzcy_prior = est_init['prob_zcx'] @ pxcy_list[midx] 
				tmp_dec_list.append(pzcy_prior)
				tmp_est_list.append(est_init['prob_zcx'])
				mizx_list.append(est_init['IXZ'])
				mizy_list.append(est_init['IYZ'])
			else:
				return {'conv':False}
		else:
			# step N: run the incremental model
			#ne = pxy_list[midx].shape[0]
			inner_loop_conv = False
			best_out = {}
			best_mic = 0.0
			for rn in range(d_retry):
				est_out = mvib_inc_single_type2(pxy_list[midx],gamma_vec[midx],convthres,maxiter,**{'prior_pzcy':pzcy_prior,**kwargs})
				# could be divergent
				if not est_out['conv']:
					# how to handle non converging cases?
					# could try some more time...
					print('ERROR: view {:>5} failed (retry count:{:>3})'.format(midx,rn))
					#return {'conv':False}
				else:
					if est_out['IZCY']>best_mic:
						best_out = copy.deepcopy(est_out)
						best_mic = est_out['IZCY']
					inner_loop_conv=True
					print('LOG: view {:>5} converged (retry count:{:>3}): best_IYZC={:>10.5f}'.format(midx,rn,best_mic))
			if not inner_loop_conv:
				return {'conv':False}
			else:
				print("LOG:incremental view {:>5} converged--IXZ|Z'={:>10.4f}, IYZ|Z'={:>10.4f}".format(midx,best_out['IZCX'],best_out['IZCY']))
				# accumulate the backward encoder
				pzcy_prior = best_out['pzzcy'] # the backward encoder for next iteration # NOTE: is a joint probability
				tmp_est_list.append(best_out['pzczx']) # this is conditional marginal prob
				tmp_dec_list.append(best_out['pzzcy']) # this is conditional joint prob
				mizx_list.append(best_out['IZCX'])
				mizy_list.append(best_out['IZCY'])

	# FIXME: for debugging purpose
	print('LOG:incremental method converged!')
	# put the encoders back in order
	est_list  =[None]*nview
	dec_list  =[None]*nview
	out_mizx_list =[0.0]*nview
	out_mizy_list =[0.0]*nview
	for idx in range(nview):
		est_list[mi_idx_sortlist[idx]] = tmp_est_list[idx]
		dec_list[mi_idx_sortlist[idx]] = tmp_dec_list[idx]
		out_mizx_list[mi_idx_sortlist[idx]] = mizx_list[idx]
		out_mizy_list[mi_idx_sortlist[idx]] = mizy_list[idx]
	return {'conv':True,'enc_list':est_list,'dec_list':dec_list,
			'IXZ_list':out_mizx_list,'IYZ_list':out_mizy_list}

#def mvib_inc_single_type1(pxy,gamma,nz,convthres,maxiter,**kwargs):
#	return None


def mvib_inc_single_type2(pxy,gamma,convthres,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	penc = kwargs['penalty_coefficient']
	ssinit = kwargs['init_stepsize']
	sscale = kwargs['stepsize_scale']
	(nx,ny) = pxy.shape
	py = np.sum(pxy,axis=0)
	px = np.sum(pxy,axis=1)
	nz = px.shape[0]
	pxcy = pxy / py[None,:]
	pycx = (pxy/px[:,None]).T
	
	# priors: p(z'|y), assume it is given, a tensor
	prior_pzcy = kwargs['prior_pzcy']
	prior_pz = np.sum(prior_pzcy*py[...,:],axis=-1)
	#prior_pzcx = np.matmul(prior_pzcy,pycx)  # should be (...,nx)
	# equivalently:
	prior_pzcx = prior_pzcy@pycx
	# pre computed constants
	const_grad_x_scalar = prior_pzcx * px[...,:]
	const_grad_y_scalar = prior_pzcy * py[...,:]
	const_gradx_pendz = (prior_pzcx * px[...,:])/prior_pz[...,None]   #p(x|z') # shape: z',x
	# primal variables

	# FIXME:
	# start with random initialization, could use deterministic start for speed and boundary points
	tmp_pzcx_shape = tuple([nz]+list(prior_pzcx.shape))
	var_pzcx = rs.rand(*tmp_pzcx_shape)
	var_pzcx /= np.sum(var_pzcx,axis=0)

	var_pz = np.sum( (prior_pzcx*px[...,:])[None,...]*var_pzcx,axis=-1)
	var_pz /= prior_pz[None,...]
	'''
	print(np.sum(var_pzcx,axis=0))
	print(np.sum(var_pzcx,axis=0))
	print(np.sum(prior_pzcx,axis=0))
	print(np.sum(prior_pzcy,axis=0))
	print(np.sum(pxcy,axis=0))
	'''
	#var_pzcy = (var_pzcx*prior_pzcx[None,...])@pxcy/prior_pzcy[None,]
	var_pzcy = var_pzcx @ pxcy
	# augmented variables
	dual_z = np.zeros(var_pz.shape)
	dual_zy= np.zeros(var_pzcy.shape)
	# system counters
	itcnt = 0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# calculate the error
		errz = np.sum((const_grad_x_scalar)[None,:]*var_pzcx,axis=-1)/prior_pz[None,...] - var_pz
		errzy = var_pzcx@pxcy - var_pzcy
		# gradient of x
		# -gamma H(Z|Z',X)
		grad_x = gamma*(np.log(var_pzcx)+1)*const_grad_x_scalar[None,:] + (dual_z + penc * errz)[...,None] * const_gradx_pendz\
				+(dual_zy + penc*errzy)@pxcy.T # NOTE that, approximate p(x|z'z,y) \approx p(x|y)
		mean_grad_x = grad_x - np.mean(grad_x,axis=0)
		ss_x = gd.naiveStepSize(var_pzcx,-mean_grad_x,ssinit,sscale)
		if ss_x == 0:
			break
		new_var_pzcx = var_pzcx - ss_x * mean_grad_x

		# re calculate the error
		errz = np.sum((const_grad_x_scalar)[None,:]*new_var_pzcx,axis=-1)/prior_pz[None,...] - var_pz
		errzy = new_var_pzcx@pxcy - var_pzcy
		# gradient of z
		# (gamma - 1) H(Z,Z')
		grad_z = (1-gamma)*(np.log(var_pz)+1) * prior_pz[None,...] -(dual_z + penc * errz)
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		ss_z = gd.naiveStepSize(var_pz,-mean_grad_z,ssinit,sscale)
		if ss_z ==0:
			break
		# gradient of y
		# H(Z|Z'Y)
		grad_y = -(np.log(var_pzcy)+1) * const_grad_y_scalar[None,...] - (dual_zy+penc * errzy)
		mean_grad_y = grad_y - np.mean(grad_y,axis=0)
		ss_y = gd.naiveStepSize(var_pzcy,-mean_grad_y,ss_z,sscale)
		if ss_y == 0:
			break
		new_var_pz = var_pz - mean_grad_z * ss_y
		new_var_pzcy = var_pzcy - mean_grad_y * ss_y
		# dual update
		# update the error again
		errz = np.sum((const_grad_x_scalar)[None,...]*new_var_pzcx,axis=-1)/prior_pz[None,...] - new_var_pz
		errzy = new_var_pzcx@pxcy - new_var_pzcy
		dual_z +=  penc*errz
		dual_zy +=  penc*errzy
		# convergence criterion
		convz = 0.5*np.sum(np.fabs(errz),axis=0)
		convzy = 0.5*np.sum(np.fabs(errzy),axis=0)
		if np.all(np.array(convz < convthres)) and np.all(np.array(convzy<convthres)):
			conv_flag = True
			break
		else:
			var_pzcx = new_var_pzcx
			var_pz = new_var_pz
			var_pzcy = new_var_pzcy
		# pre-calculate the backward encoder
	out_backward_enc = var_pzcy * prior_pzcy[None,...]
	# calculate the conditional mutual information, might need to use masking scale to prevent underflow
	joint_pzcx = var_pzcx * const_grad_x_scalar
	mic_zcx = np.sum(joint_pzcx*np.log(var_pzcx/var_pz[...,None]))
	joint_pzcy = var_pzcy * const_grad_y_scalar
	mic_zcy = np.sum(joint_pzcy*np.log(var_pzcy/var_pz[...,None]))

	return {'pzczx':var_pzcx,'pzzcy':out_backward_enc,'niter':itcnt,'conv':conv_flag,'IZCX':mic_zcx,'IZCY':mic_zcy}
# compared algorithms

def ib_orig(pxy,convthres,gamma,maxiter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	beta = 1/gamma
	(nx,ny) = pxy.shape
	px = np.sum(pxy,axis=1)
	nz = px.shape[0]
	py = np.sum(pxy,axis=0)
	pycx = np.transpose(np.diag(1./px)@pxy)
	pxcy = pxy@np.diag(1./py)
	# on IB, the initialization matters
	# use random start (*This is the one used for v2)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pz = pzcx @ px

	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	
	# ready to start
	itcnt = 0
	conv_flag = False
	while itcnt<maxiter:
		# compute ib kernel
		new_pzcx= np.zeros((nz,nx))
		kl_oprod = np.expand_dims(1./pycz,axis=-1)@np.expand_dims(pycx,axis=1)
		kl_ker = np.sum(np.repeat(np.expand_dims(pycx,axis=1),nz,axis=1)*np.log(kl_oprod),axis=0)
		new_pzcx = np.diag(pz)@np.exp(-beta*kl_ker)

		# standard way, normalize to be valid probability.
		new_pzcx /=np.sum(new_pzcx,axis=0)
		itcnt+=1
		# total variation convergence criterion
		diff = 0.5* np.sum(np.fabs(new_pzcx-pzcx))
		if diff < convthres:
			conv_flag = True
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
	mixz = ut.calcMI(pzcx*px[None,:])
	miyz = ut.calcMI(pycz*pz[None,:])
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'conv':conv_flag}


def varinf_ba2v(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	# this approach is a variational method
	# update the encoder and decoder in an alternating fashion
	# this reference approach only support 2 views for now
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pxy_v1 = pxy_list[0]
	pxy_v2 = pxy_list[1]
	px_v1 = np.sum(pxy_v1,axis=1)
	pxcy_v1 = pxy_v1 / px_v1[:,None]

	py = np.sum(pxy_v1,axis=0)
	ny = len(py)
	(nx_v1, _) = pxy_v1.shape
	(nx_v2, _) = pxy_v2.shape
	
	px_v2 = np.sum(pxy_v2,axis=1)
	pxcy_v2 = pxy_v2 / px_v2[:,None]

	pycx_v1 = (pxy_v1/py[None,:]).T
	pycx_v2 = (pxy_v2/py[None,:]).T

	gamma_v1 = gamma_vec[0]
	gamma_v2 = gamma_vec[1]
	
	nz_v1 = nx_v1
	nz_v2 = nx_v2
	
	pzcx_v1 = rs.rand(nx_v1,nx_v1)
	pzcx_v1 /= np.sum(pzcx_v1,axis=0)
	pzcx_v2 = rs.rand(nx_v2,nx_v2)
	pzcx_v2 /= np.sum(pzcx_v2,axis=0)
	pz_v1 = pzcx_v1 @ px_v1
	pz_v2 = pzcx_v2 @ px_v2

	# variational parameters
	qzcy_v1 = pzcx_v1 @ pxcy_v1
	qzcy_v2 = pzcx_v2 @ pxcy_v2
	qz_v1 = pzcx_v1 @ px_v1
	qz_v2 = pzcx_v2 @ px_v2
	qyz12 = py[:,...] * np.expand_dims(qzcy_v1.T,axis=-1) * np.expand_dims(qzcy_v2.T,axis=1)
	print(np.sum(qyz12)) # =1.5?????
	sys.exit()
	qz12 = np.sum(qyz12,axis=0)
	qy_cz12 = qyz12 / np.sum(qyz12,axis=0)

	itcnt = 0
	conv_flag = False
	while itcnt< maxiter:
		itcnt+=1
		# prepare decoders
		pzcy_v1 = pzcx_v1 @ pxcy_v1
		pzcy_v2 = pzcx_v2 @ pxcy_v2
		# calculate the kl kernel
		ker_v1 = np.zeros((nz_v1,nx_v1))
		for idxz1 in range(nz_v1):
			for idxx1 in range(nx_v1):
				tmpsum1 = 0
				tmpsum2 = 0
				for idxz2 in range(nx_v2):
					for idxy in range(ny):
						tmpsum1 += pycx_v1[idxy,idxx1]*pzcy_v2[idxz2,idxy]*np.log(qy_cz12[idxy,idxz1,idxz2])
						tmpsum2 += pycx_v1[idxy,idxx1]*pzcy_v2[idxz2,idxy]*np.log(qz12[idxz1,idxz2])
				ker_v1[idxz1,idxx1] = 1/gamma_v1 * tmpsum1 + tmpsum2
		new_pzcx_v1 = np.exp(ker_v1)
		new_pzcx_v1 /= np.sum(new_pzcx_v1,axis=0)
		ker_v2 = np.zeros((nz_v2,nx_v2))
		for idxz2 in range(nz_v2):
			for idxx2 in range(nx_v2):
				tmpsum1 = 0
				tmpsum2 = 0
				for idxz1 in range(nx_v1):
					for idxy in range(ny):
						tmpsum1 += pycx_v2[idxy,idxx2]*pzcy_v1[idxz1,idxy]*np.log(qy_cz12[idxy,idxz1,idxz2])
						tmpsum2 += pycx_v2[idxy,idxx2]*pzcy_v1[idxz1,idxy]*np.log(qz12[idxz1,idxz2])
				ker_v2[idxz2,idxx2] = 1/gamma_v2*tmpsum1 + (gamma_v1/gamma_v2)*tmpsum2+np.log(qz_v2[idxz2])-(gamma_v1/gamma_v2)*np.log(pz_v2[idxz2])
		new_pzcx_v2 = np.exp(ker_v2)
		new_pzcx_v2 /= np.sum(new_pzcx_v2,axis=0)
		# update the variational probabilities
		new_pz_v1 = new_pzcx_v1 @ px_v1
		new_pzcy_v1 = new_pzcx_v1 @ pxcy_v1

		new_pz_v2 = new_pzcx_v2 @ px_v2
		new_pzcy_v2 = new_pzcx_v2 @ pxcy_v2

		new_pyz12 = py[:,...] * np.expand_dims(new_pzcy_v1.T,axis=-1) * np.expand_dims(new_pzcy_v2.T,axis=1)

		new_pz12 = np.sum(new_pyz12,axis=0)

		# convergence check
		dtv1 = np.sum(np.fabs(new_pzcx_v1- pzcx_v1),axis=0)
		dtv2 =np.sum(np.fabs(new_pzcx_v2 - pzcx_v2),axis=0)
		if np.all(dtv1<convthres) and np.all(dtv2<convthres):
			conv_flag = True
			break
		else:
			# copy the update for next round
			pzcx_v1 = new_pzcx_v1
			pzcx_v2 = new_pzcx_v2
			pz_v1 = new_pz_v1
			pz_v2 = new_pz_v2
			# permutation to store the variational approximation
			qyz12 = new_pyz12
			qz12 = new_pz12
			qy_cz12 = qyz12 / np.sum(qyz12,axis=0)
	# calculate I(Y;Z_1,Z_2), from q(y,z1,z2)
	print("joint yz12:",np.sum(qy_cz12,axis=0))
	mi_yz12 = 0
	for yidx in range(ny):
		for z1idx in range(nz_v1):
			for z2idx in range(nz_v2):
				mi_yz12 += qyz12[yidx,z1idx,z2idx] * np.log(qy_cz12[yidx,z1idx,z2idx]/py[yidx])	
	# calculate I(X_1;Z_1), from pzcx_v1
	mizx_v1 = np.sum(pzcx_v1*px_v1[None,:] * np.log(pzcx_v1/pz_v1[:,None]))
	# calculate I(X_2;Z_2), from pzcx_v2
	mizx_v2 = np.sum(pzcx_v2*px_v2[None,:] * np.log(pzcx_v2/pz_v2[:,None]))
	enc_list= [pzcx_v1, pzcx_v2]
	mizx_list = [mizx_v1,mizx_v2]
	# compute the mutual information required for comparison
	return {'conv':conv_flag,'enc_list':enc_list,'niter':itcnt,'IXZ_list':mizx_list,'IZCY':mi_yz12}