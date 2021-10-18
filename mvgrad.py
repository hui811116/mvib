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
'''
def funcQObj(gamma_vec,px_list,pxcy_list,py,pcoeff):
	def valObj(pzcx_list,pz,pzcy,muz_list,muzcy_list):
		# calculating errors
		errz_list = [pz-pzcx_list[i]@px_list[i] for i in range(len(px_list))]
		errzcy_list = []
		for i in range(len(gamma_vec)):
			errzcy_list.append(pzcy-pzcx_list[i]@pxcy_list[i])
		# calculate the func values
		errznorm2 = sum([np.linalg.norm(i,2)**2 for i in errz_list])
		errzcynorm2=sum([np.linalg.norm(i,2)**2 for i in errzcy_list])
		errzddot  = sum([np.dot(muz_list[i],errz_list[i]) for i in range(len(px_list))])
		errzcyddot = sum([np.dot(muzcy_list[i],errzcy_list[i]) for i in range(len(px_list))])
		return (-1+np.sum(gamma_vec))*ut.calcEnt(pz)+ut.calcCondEnt(pzcy,py)\
				+errzddot+errzcyddot+0.5*pcoeff*(errznorm2+errzcynorm2) # FIXME, the dimension is wrong
	return valObj

def gradQObj(gamma_vec,px_list,pxcy_list,pcoeff):
	def gradObj(pzcx_list,pz,pzcy,muz_list,muzcy_list):
		errz_list = [pz-pzcx_list[i]@px_list[i] for i in range(len(px_list))]
		errzcy_list=[pzcy-pzcx_list[i]@pxcy_list[i] for i in range(len(px_list))]
		grad = (1-np.sum(gamma_vec))*(np.log(pz)+1)-(np.log(pzcy)+1)*py[None,:]\
				+sum(muz_list)+sum(muzcy_list)+pcoeff*(sum(errz_list)+sum(errzcy_list))
		return grad
	return gradObj
'''
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
	def gradObj(pzcx,pz,pzcy,pzeccx,pzec,pzeccy,mu_z,mu_zcy,mu_zeccx,mu_zec,mu_zeccy):
		errz = np.sum(pzcx*px[None,:],axis=1)-pz
		errzcy = pzcx@pxcy-pzcy
		errzec = pzcx - np.sum(pzeccx,axis=0)
		errzec_z = np.sum(pzcx*px[None,:],axis=1)-np.sum(pzec,axis=0)
		errzec_zcy=pzcx@pxcy - np.sum(pzeccy,axis=0)
		tmp_grad = gamma*(np.log(pzcx)+1)*px[None,:] + (mu_z+ pcoeff*errz)[:,None]*px[None,:]\
					 +(mu_zcy + pcoeff*errzcy)@pxcy.T\
					 +(mu_zeccx + pcoeff*errzec) \
					 +(mu_zec + pcoeff*errzec_z)[:,None]*px[None,:]\
					 +(mu_zeccy + pcoeff*errzec_zcy)@pxcy.T
		return tmp_grad
	return gradObj

# NOTE: involved common variables only
def gradPzComnObj(gamma_vec,px_list,pcoeff):
	def gradObj(pz,pzcx_list,muz_list):
		errz_list = [np.sum(pzcx_list[idx]*(px_list[idx])[None,:],axis=1)-pz for idx in range(len(px_list))]
		return (-1+np.sum(gamma_vec))*(np.log(pz)+1)-sum(muz_list)-pcoeff*sum(errz_list)
	return gradObj

# NOTE: involved common variables only
def gradPzcyComnObj(pxcy_list,py,pcoeff):
	def gradObj(pzcy,pzcx_list,muzcy_list):
		errzcy_list  =[pzcx_list[idx]@pxcy_list[idx]-pzcy for idx in range(len(pxcy_list))]
		return -(np.log(pzcy)+1)*py[None,:]-sum(muzcy_list)-pcoeff*sum(errzcy_list)
	return gradObj

def gradPzcxCmplObj(gamma,px,pcoeff):
	def gradObj(pzcx,pzeccx,mu_zeccx):
		errzeccx = pzcx - np.sum(pzeccx,axis=0)
		tmp_pzccx = np.sum(pzeccx,axis=0)
		phi_grad = -gamma * px[None,None,:]*(-np.log(pzeccx)-1+np.log(tmp_pzccx)[None,...]+pzeccx/tmp_pzccx[None,...])
		return phi_grad - mu_zeccx[None,...]-pcoeff*errzeccx[None,...]
	return gradObj

def gradPzCmplObj(gamma,px,pcoeff):
	def gradObj(pzcx,pzec,mu_zec):
		errzec = np.sum(pzcx*px[None,:],axis=1) - np.sum(pzec,axis=0)
		tmp_pzc = np.sum(pzec,axis=0)
		psi_z_grad = (gamma-1)*(-np.log(pzec)-1+np.log(tmp_pzc)[None,:]+pzec/tmp_pzc[None,:])
		return psi_z_grad - mu_zec[None,:]- pcoeff*errzec[None,:]
	return gradObj
def gradPzcyCmplObj(pxcy,py,pcoeff):
	def gradObj(pzcx,pzeccy,mu_zeccy):
		errzeccy = pzcx@pxcy - np.sum(pzeccy,axis=0)
		tmp_pzccy = np.sum(pzeccy,axis=0)
		psi_zcy_grad = py[None,None,:] * (-np.log(pzeccy)-1+np.log(tmp_pzccy)[None,...]+pzeccy/tmp_pzccy[None,...])
		return psi_zcy_grad - mu_zeccy[None,...] - pcoeff*errzeccy[None,...]
	return gradObj

# implement the naive step size search algorithm
def naiveStepSize(prob,update,init_step,scale):
	stepsize= init_step
	while np.any(prob+stepsize*update<=0.0) or np.any(prob+stepsize*update>=1.0):
		stepsize *= scale
		if stepsize< 1e-9:
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
