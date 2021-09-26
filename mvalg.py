import numpy as np
import mvgrad as gd
import mvutils as ut
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import sys

'''
def mvib_2v(pxy_list,gamma_vec,nz,convthres,maxiter,**kwargs):
	assert len(pxy_list) == 2
	pxy1 = pxy_list[0]
	pxy2 = pxy_list[1]
	assert gamma_vec.shape[0] == 2
	gamma1 = gamma_vec[0]
	gamma2 = gamma_vec[1]
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	pen_coeff = kwargs['penalty_coefficient']
	gd_ss_init = kwargs['init_stepsize']
	gd_ss_scale = kwargs['stepsize_scale']
	# preparation for initialization
	(nx1,ny1) = pxy1.shape
	(nx2,ny2) = pxy2.shape
	assert ny1==ny2, "the dimension of y1 and y2 must be the same!"
	ny = ny1

	# the marginal probability
	# FIXME: right now, assume the marginal py are the same in both pxy1 pxy2
	py = np.sum(pxy1,0)
	px_v1 = np.sum(pxy1,1)
	px_v2 = np.sum(pxy2,1)
	px_list = [px_v1,px_v2]
	pxcy_v1 =  pxy1 / py[None,:]
	pxcy_v2 = pxy2 / py[None,:]
	pxcy_list = [pxcy_v1,pxcy_v2]
	gamma_vec = np.array([gamma1,gamma2])

	# initialize with random values
	pzcx_v1 = rs.rand(nz,nx1)
	pzcx_v1 /= np.sum(pzcx_v1,axis=0)
	
	pzcx_v2 = rs.rand(nz,nx2)
	pzcx_v2 /= np.sum(pzcx_v2,axis=0)

	# augmented variables
	# FIXME: Can use averaging as a initial point
	pz = 0.5 * (pzcx_v1@px_v1 + pzcx_v2@px_v2)
	pz /= np.sum(pz)

	pzcy = 0.5 *(pzcx_v1@pxcy_v1 + pzcx_v2@pxcy_v2)
	pzcy /= np.sum(pzcy,axis=0)

	# dual variables
	dz_v1 = np.zeros(nz)
	dz_v2 = np.zeros(nz)
	dzcy_v1 = np.zeros((nz,ny))
	dzcy_v2 = np.zeros((nz,ny))

	# objects for gradients
	fobj_pzcx_v1 = gd.funcPzcxObj(gamma1,px_v1,pxcy_v1,pen_coeff)
	fobj_pzcx_v2 = gd.funcPzcxObj(gamma2,px_v2,pxcy_v2,pen_coeff)
	gobj_pzcx_v1 = gd.gradPzcxObj(gamma1,px_v1,pxcy_v1,pen_coeff)
	gobj_pzcx_v2 = gd.gradPzcxObj(gamma2,px_v2,pxcy_v2,pen_coeff)

	fobj_pz      = gd.funcPzObj(gamma_vec,px_list,pen_coeff)
	fobj_pzcy    = gd.funcPzcyObj(pxcy_list,py,pen_coeff)
	gobj_pz      = gd.gradPzObj(gamma_vec,px_list,pen_coeff)
	gobj_pzcy    = gd.gradPzcyObj(pxcy_list,py,pen_coeff)

	# counters and flags
	itcnt = 0
	flag_conv = False
	while itcnt < maxiter:
		itcnt += 1
		# update the first view enc
		grad_pzcx_v1 = gobj_pzcx_v1(pzcx_v1,pz,pzcy,dz_v1,dzcy_v1)
		# manually mean
		mean_grad_pzcx_v1 = grad_pzcx_v1 - np.mean(grad_pzcx_v1,axis=0)
		ss_zcx_v1 = gd.naiveStepSize(pzcx_v1,-mean_grad_pzcx_v1,gd_ss_init,gd_ss_scale)
		if ss_zcx_v1 ==0:
			break
		#ss_zcx_v1 = gd.armijoStepSize(pzcx_v1,-mean_grad_pzcx_v1,gd_ss)
		new_pzcx_v1 = pzcx_v1 -ss_zcx_v1 * mean_grad_pzcx_v1
		# update the second view enc
		grad_pzcx_v2 = gobj_pzcx_v2(pzcx_v2,pz,pzcy,dz_v2,dzcy_v2)
		mean_grad_pzcx_v2 = grad_pzcx_v2 - np.mean(grad_pzcx_v2,axis=0)
		ss_zcx_v2 = gd.naiveStepSize(pzcx_v2,-mean_grad_pzcx_v2,gd_ss_init,gd_ss_scale)
		if ss_zcx_v2 ==0:
			break
		new_pzcx_v2 = pzcx_v2 - ss_zcx_v2* mean_grad_pzcx_v2
		new_pzcx_list = [new_pzcx_v1,new_pzcx_v2]
		# update the augmented var
		grad_pz = gobj_pz(new_pzcx_list,pz,[dz_v1,dz_v2])
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale)
		if ss_z == 0:
			break
		new_pz = pz -ss_z * mean_grad_pz
		grad_pzcy = gobj_pzcy(new_pzcx_list,pzcy,[dzcy_v1,dzcy_v2])
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		new_pzcy = pzcy - mean_grad_pzcy * ss_zcy
		# update the dual variables
		err_z_v1   = new_pz   - new_pzcx_v1 @ px_v1
		err_z_v2   = new_pz   - new_pzcx_v2 @ px_v2
		err_zcy_v1 = new_pzcy - new_pzcx_v1 @ pxcy_v1
		err_zcy_v2 = new_pzcy - new_pzcx_v2 @ pxcy_v2
		dz_v1   = dz_v1   + pen_coeff*err_z_v1
		dz_v2   = dz_v2   + pen_coeff*err_z_v2
		dzcy_v1 = dzcy_v1 + pen_coeff*err_zcy_v1
		dzcy_v2 = dzcy_v2 + pen_coeff*err_zcy_v2
		# convergence criterion
		conv_z_v1   = 0.5*np.sum(np.fabs(err_z_v1))<convthres
		conv_z_v2   = 0.5*np.sum(np.fabs(err_z_v2))<convthres
		conv_zcy_v1 = np.all(0.5*np.sum(np.fabs(err_zcy_v1),axis=0)<convthres)
		conv_zcy_v2 = np.all(0.5*np.sum(np.fabs(err_zcy_v2),axis=0)<convthres)
		if conv_z_v1 and conv_z_v2 and conv_zcy_v1 and conv_zcy_v2:
			flag_conv = True
			break
		# update the registers
		pzcx_v1 = new_pzcx_v1
		pzcx_v2 = new_pzcx_v2
		pz = new_pz
		pzcy = new_pzcy
	return {'pzcx_list':[pzcx_v1,pzcx_v2],'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv}
'''

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

		# FIXME: this part actually can be parallelly computed
		# TODO: understand python parallelism. Multprocessing?
		_ss_interrupt = False
		new_pzcx_list = [np.zeros((nz,i.shape[0])) for i in pxy_list]
		for i in range(nview):
			tmp_grad_pzcx = gobj_pzcx_list[i](pzcx_list[i],pz,pzcy,dz_list[i],dzcy_list[i])
			mean_tmp_grad_pzcx = tmp_grad_pzcx - np.mean(tmp_grad_pzcx,axis=0)[None,:]
			tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[i],-mean_tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzcx==0:
				_ss_interrupt = True
				break
			# TODO: Armijo stepsize update
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
		'''
		# armijo condition
		ss_z = gd.armijoStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale,1e-4,fobj_pz,gobj_pz,\
					**{'pzcx_list':new_pzcx_list,'muz_list':dz_list})
		if ss_z == 0:
			break
		'''
		grad_pzcy = gobj_pzcy(pzcy,new_pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)[None,:]
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		# armijo condition
		'''
		ss_zcy = gd.armijoStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale,1e-4,fobj_pzcy,gobj_pzcy,\
					**{'pzcx_list':new_pzcx_list,'muzcy_list':dzcy_list})
		if ss_zcy == 0:
			break
		'''
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
	'''
	errz_list  = [new_pz - item@px_list[idx] for idx,item in enumerate(new_pzcx_list)]
	errzcy_list= [new_pzcy- item@pxcy_list[idx] for idx,item in enumerate(new_pzcx_list)]
	print('zlist_conv:')
	print(errz_list)
	print('zylist_conv:')
	print(errzcy_list)
	'''
	return {'pzcx_list':pzcx_list,'pz':pz,'pzcy':pzcy,'niter':itcnt,'conv':flag_conv}



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
