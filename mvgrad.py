import numpy as np
import mvutils as ut
import sys


# implemented as gradient descent algorithms
# Object version of the function value and gradients
def funcPzcxObj(gamma,px,pxcy,pcoeff):
	def valObj(pzcx,pz,pzcy,mu_z,mu_zcy):
		err_z = pz-np.sum(pzcx * px[None,:],axis=1)
		err_zcy=pzcy -pzcx@pxcy
		return -gamma * ut.calcCondEnt(pzcx,px)+np.sum(mu_z*err_z)+np.linalg.norm(err_z,2)**2 \
				+np.sum(mu_zcy*err_zcy)+np.linalg.norm(err_zcy,2)**2
	return valObj

def gradPzcxObj(gamma,px,pxcy,pcoeff):
	# note: this returns the raw gradient. might need additional processing to assure a probability dist.
	def gradObj(pzcx,pz,pzcy,mu_z,mu_zcy):
		err = pz - np.sum(pzcx * px[None,:],axis=1)
		errzy = pzcy - pzcx @ pxcy
		grad = gamma * (np.log(pzcx)+1)*px[None,:]-(mu_z+pcoeff*err)[:,None]*px[None,:]\
				-(mu_zcy+pcoeff*(errzy))@pxcy.T
		return grad
	return gradObj

# NOTE: the following returns Pz func, grad separately
def funcPzObj(gamma_vec,px_list,pcoeff):
	def valObj(pz,pzcx_list,muz_list):
		errz_list = [pz-np.sum(pzcx_list[i]*(px_list[i])[None,:],axis=1) for i in range(len(px_list))]
		errznorm2 = sum([np.linalg.norm(i,2)**2 for i in errz_list])
		errzddot  = sum([np.dot(muz_list[i],errz_list[i]) for i in range(len(px_list))])
		return (-1+np.sum(gamma_vec))*ut.calcEnt(pz)+errzddot+0.5*pcoeff*errznorm2
	return valObj

def gradPzObj(gamma_vec,px_list,pcoeff):
	def gradObj(pz,pzcx_list,muz_list):
		errz_list = [pz-np.sum(pzcx_list[i]*(px_list[i])[None,:],axis=1) for i in range(len(px_list))]
		return (1-np.sum(gamma_vec))*(np.log(pz)+1)+sum(muz_list)+pcoeff*sum(errz_list)
	return gradObj
# NOTE: the following returns Pz|y func, grad separately
def funcPzcyObj(pxcy_list,py,pcoeff):
	def funcObj(pzcy,pzcx_list,muzcy_list):
		errzcy_list = [pzcy-pzcx_list[i]@pxcy_list[i] for i in range(len(pxcy_list))]
		errzcynorm2 = sum([np.linalg.norm(i,2)**2 for i in errzcy_list])
		errzcyddot = sum([np.sum(muzcy_list[i]*errzcy_list[i]) for i in range(len(pxcy_list))])
		return ut.calcCondEnt(pzcy,py)+errzcyddot+0.5*pcoeff*errzcynorm2
	return funcObj
def gradPzcyObj(pxcy_list,py,pcoeff):
	def gradObj(pzcy,pzcx_list,muzcy_list):
		errzcy_list = [pzcy-pzcx_list[i]@pxcy_list[i] for i in range(len(pxcy_list))]
		return -(np.log(pzcy)+1)*py[None,:]+sum(muzcy_list)+pcoeff*sum(errzcy_list)
	return gradObj

# common-complement gradients
# NOTE: the complement penalty contains augmented variables for the common representation
#       and therefore introces extra proximal terms. Should be careful in handling the tensor gradients.
# the Comn stands for the common information
# the Cmpl stands for the complement information

# NOTE: should involve complement view as well
def gradPzcxComnObj(gamma,px,pxcy,pcoeff):
	def gradObj(pzcx,pz,pzcy,pzeccx,mu_z,mu_zcy,mu_zec):
		errz = np.sum(pzcx*px[None,:],axis=1)-pz
		errzcy = pzcx@pxcy-pzcy
		errzec = pzcx - np.sum(pzeccx,axis=0)
		tmp_grad = gamma*(np.log(pzcx)+1)*px[None,:] + (mu_z+ pcoeff*errz)[:,None]*px[None,:]\
					 +(mu_zcy + pcoeff*errzcy)@pxcy.T\
					 +(mu_zec + pcoeff*errzec)
		return tmp_grad
	return gradObj

# NOTE: involved common variables only
def gradPzComnObj(gamma_vec,px_list,pcoeff):
	def gradObj(pz,pzcx_list,muz_list):
		errz_list = [np.sum(pzcx_list[idx]*(px_list[idx])[None,:],axis=1)-pz for idx in range(len(px_list))]
		return (1-np.sum(gamma_vec))*(np.log(pz)+1)-sum(muz_list)-pcoeff*sum(errz_list)
	return gradObj

# NOTE: involved common variables only
def gradPzcyComnObj(pxcy_list,py,pcoeff):
	def gradObj(pzcy,pzcx_list,muzcy_list):
		errzcy_list  =[pzcx_list[idx]@pxcy_list[idx]-pzcy for idx in range(len(pxcy_list))]
		return -(np.log(pzcy)+1)*py[None,:]-sum(muzcy_list)-pcoeff*sum(errzcy_list)
	return gradObj

def gradPzcxCmplObj(gamma,px,pxcy,pycx,pcoeff):
	def gradObj(pzeccx,aug_pzcx,mu_zec):
		errzec = aug_pzcx - np.sum(pzeccx,axis=0)
		pzcx = np.sum(pzeccx,axis=0)
		pz = np.sum(pzcx * px,axis=1)
		pzcy = pzcx@pxcy
		pzec = np.sum(pzeccx*px[...,:],axis=2)
		pzeccy = pzeccx@pxcy
		tmp_cmpl = np.log(pzeccx) - np.log(pzec)[...,None] - (1/gamma)*(np.log(pzeccy)-np.log(pzec)[...,None])@pycx
		tmp_cmon = np.log(pzcx) - np.log(pz)[:,None] - (1/gamma)*(np.log(pzcy)-np.log(pz)[:,None])@pycx
		tmp_diff = (pzec/pz)[...,None]-pzeccx/pzcx-(1/gamma)*( (pzec/pz)[...,None]-(pzeccy/pzcy)@pycx)
		return gamma*px[...,:]*(tmp_cmpl-tmp_cmon+tmp_diff) - (mu_zec+pcoeff*errzec)[None,...]
	return gradObj

# masking gradient to avoid underflowing
def maskGradPzcxCmonObj(gamma,px,pxcy,pcoeff,mask_thres):
	def gradObj(pzcx,pz,pzcy,pzeccx,mu_z,mu_zcy,mu_zec):
		errzec = pzcx - np.sum(pzeccx,axis=0)
		err = np.sum(pzcx * px[None,:],axis=1)-pz
		errzy = pzcx @pxcy- pzcy
		val_mask= np.logical_and(pzcx>mask_thres,pzcx < 1-mask_thres)
		raw_grad = np.zeros(pzcx.shape)
		all_grad = gamma * (np.log(pzcx)+1)*px[None,:]+(mu_z+pcoeff*err)[:,None]*px[None,:]\
								+(mu_zcy+pcoeff*errzy)@pxcy.T\
								+(mu_zec + pcoeff*errzec)
		raw_grad[val_mask] = all_grad[val_mask]
		# return the mean subtracted gradient
		mcount = np.any(val_mask,axis=0)
		for idx in range(len(mcount)):
			if mcount[idx]:
				tmpmean = np.mean(raw_grad[:,idx][val_mask[:,idx]])
				raw_grad[:,idx][val_mask[:,idx]] -= tmpmean
		return raw_grad
	return gradObj

def maskGradPzcxCmplObj(gamma,px,pxcy,pycx,pcoeff,mask_thres):
	def gradObj(pzeccx,aug_pzcx,mu_zec):
		errzec = aug_pzcx - np.sum(pzeccx,axis=0)
		pzcx = np.sum(pzeccx,axis=0)
		pz = np.sum(pzcx * px[None,:],axis=1)
		pzcy = pzcx@pxcy
		pzec = np.sum(pzeccx*px[...,:],axis=2)
		pzeccy = pzeccx@pxcy
		# masking
		cmpl_mask = np.logical_and(pzeccx>mask_thres,pzeccx<1-mask_thres) # this is the overall elements to be updated (px!=0)
		gradout = np.zeros(pzeccx.shape)
		# raw gradient
		tmp_cmpl = np.log(pzeccx/pzec[...,None]) -(1/gamma)*(np.log(pzeccy/pzec[...,None])@pycx)
		tmp_cmon = np.log(pzcx/pz[:,None])-(1/gamma)*(np.log(pzcy/pz[:,None])@pycx)
		tmp_diff = (pzec/pz[None,:])[...,None]-pzeccx/pzcx[None,...]-(1/gamma)*((pzec/pz[None,:])[...,None]-(pzeccy/pzcy[None,...])@pycx)
		all_grad = (gamma*px[...,:]*(tmp_cmpl-tmp_cmon+tmp_diff)-(mu_zec+pcoeff*errzec)[None,...])
		gradout[cmpl_mask] = all_grad[cmpl_mask]
		# subtract gradient mean conditioned on x
		cond_check = np.any(cmpl_mask,axis=(0,1))
		for idx in range(len(cond_check)):
			if cond_check[idx]:
				tmp_mean = np.mean(gradout[...,idx][cmpl_mask[...,idx]])
				gradout[...,idx][cmpl_mask[...,idx]]-= tmp_mean
		return gradout
	return gradObj


# implement the naive step size search algorithm
def naiveStepSize(prob,update,init_step,scale):
	stepsize= init_step
	while np.any(prob+stepsize*update<=0.0) or np.any(prob+stepsize*update>=1.0):
		stepsize *= scale
		if stepsize< 1e-11:
			stepsize = 0
			break
	return stepsize
# implement the armijo condition for step size search
def armijoStepSize(prob,update,init_step,scale,c1,funcObj,gradObj,**kwargs):
	stepsize = init_step
	fnext = funcObj(prob+stepsize*update,**kwargs)
	fnow = funcObj(prob,**kwargs)
	gnow = gradObj(prob,**kwargs)
	while fnext > fnow + c1 * stepsize * np.sum(update*gnow):
		if stepsize <1e-9:
			stepsize = 0
			break
		stepsize *= scale
		fnex = funcObj(prob+stepsize*update,**kwargs)
	return stepsize

def naiveConditionalStepSize(prob,update,target,init_step,scale):
	stepsize = init_step
	while np.any(prob+stepsize*update>target) or np.any(prob+stepsize*update<0):
		stepsize *= scale
		if stepsize < 1e-11:
			stepsize = 0
			break
	return stepsize
