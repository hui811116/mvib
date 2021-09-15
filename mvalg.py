import numpy as np
import mvgrad as gd
import mvutils as ut
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


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
	pz = rs.rand(nz)
	pz /= np.sum(pz)

	pzcy = rs.rand(nz,ny)
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
	pz = rs.rand(nz)
	pz /= np.sum(pz)
	pzcy = rs.rand(nz,ny)
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

		# FIXME: this part actually can be parallelly computed
		# TODO: understand python parallelism. Multprocessing?
		new_pzcx_list = [np.zeros((nz,i.shape[0])) for i in pxy_list]
		for i in range(nview):
			tmp_grad_pzcx = gobj_pzcx_list[i](pzcx_list[i],pz,pzcy,dz_list[i],dzcy_list[i])
			mean_tmp_grad_pzcx = tmp_grad_pzcx - np.mean(tmp_grad_pzcx,axis=0)
			tmp_ss_pzcx = gd.naiveStepSize(pzcx_list[i],-mean_tmp_grad_pzcx,gd_ss_init,gd_ss_scale)
			if tmp_ss_pzcx==0:
				break
			# TODO: Armijo stepsize update
			new_pzcx_list[i] = pzcx_list[i] -tmp_ss_pzcx * mean_tmp_grad_pzcx
		# update the augmented var
		grad_pz = gobj_pz(new_pzcx_list,pz,dz_list)
		mean_grad_pz = grad_pz - np.mean(grad_pz)
		ss_z = gd.naiveStepSize(pz,-mean_grad_pz,gd_ss_init,gd_ss_scale)
		if ss_z == 0:
			break
		new_pz = pz -ss_z * mean_grad_pz
		grad_pzcy = gobj_pzcy(new_pzcx_list,pzcy,dzcy_list)
		mean_grad_pzcy = grad_pzcy - np.mean(grad_pzcy,axis=0)
		ss_zcy = gd.naiveStepSize(pzcy,-mean_grad_pzcy,gd_ss_init,gd_ss_scale)
		if ss_zcy == 0:
			break
		new_pzcy = pzcy - mean_grad_pzcy * ss_zcy
		# update the dual variables
		# first, calculate the errors
		errz_list  = [new_pz - item@px_list[idx] for idx,item in enumerate(new_pzcx_list)]
		errzcy_list= [new_pzcy- item@pxcy_list[idx] for idx,item in enumerate(new_pzcx_list)]
		dz_list = [ item + pen_coeff*(errz_list[idx]) for idx,item in enumerate(dz_list)]
		dzcy_list=[ item + pen_coeff*(errzcy_list[idx]) for idx,item in enumerate(dzcy_list)]

		# convergence criterion
		conv_z_list = [0.5*np.sum(np.fabs(item))<convthres for item in errz_list]
		conv_zcy_list = [np.all(0.5*np.sum(np.fabs(item),axis=0))<convthres for item in errzcy_list]
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