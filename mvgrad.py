import numpy as np
import mvutils as ut
import sys


# implemented as gradient descent algorithms

def funcPzcxObj(gamma,px,pxcy,pcoeff):
	def valObj(pzcx,pz,pzcy,mu_z,mu_zcy):
		err_z = pz-pzcx @ px
		err_zcy=pzcy -pzcx@pxcy
		return -gamma * ut.calcCondEnt(pzcx,px)+np.sum(mu_z*err_z)+np.linalg.norm(err_z,2)**2 \
				+np.sum(mu_zcy,err_zcy)+np.linalg.norm(err_zcy,2)**2
	return valObj

def gradPzcxObj(gamma,px,pxcy,pcoeff):
	# note: this returns the raw gradient. might need additional processing to assure a probability dist.
	def gradObj(pzcx,pz,pzcy,mu_z,mu_zcy):
		err = pz - pzcx @ px
		errzy = pzcy - pzcx @ pxcy
		grad = gamma * (np.log(pzcx)+1)*px[None,:]-(mu_z+pcoeff*err)[:,None]@px[None,:]\
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
	def valObj(pzcx_list,pz,muz_list):
		errz_list = [pz-pzcx_list[i]@px_list[i] for i in range(len(px_list))]
		errznorm2 = sum([np.linalg.norm(i,2)**2 for i in errz_list])
		errzddot  = sum([np.dot(muz_list[i],errz_list[i]) for i in range(len(px_list))])
		return (-1+np.sum(gamma_vec))*ut.calcEnt(pz)+errzddot+0.5*pcoeff*errznorm2
	return valObj

def gradPzObj(gamma_vec,px_list,pcoeff):
	def gradObj(pzcx_list,pz,muz_list):
		#print('debugging')
		#print(muz_list)
		#print(sum(muz_list))                # this is correct
		#print(np.sum(np.array(muz_list),0)) # this is correct
		#print(np.sum(np.array(muz_list),1)) # this is wrong
		errz_list = np.array([pz-pzcx_list[i]@px_list[i] for i in range(len(px_list))])
		return (1-np.sum(gamma_vec))*(np.log(pz)+1)+np.sum(muz_list,0)+pcoeff*sum(errz_list)
	return gradObj
# NOTE: the following returns Pz|y func, grad separately
def funcPzcyObj(pxcy_list,py,pcoeff):
	def funcObj(pzcx_list,pzcy,muzcy_list):
		errzcy_list = np.array([pzcy-pzcx_list[i]@pxcy_list[i] for i in range(len(pxcy_list))])
		errzcynorm2 = np.sum(np.array([np.linalg.norm(i,2)**2 for i in errzcy_list]))
		errzcyddot = np.sum(np.array([np.sum(muzcy_list[i]*errzcy_list[i]) for i in range(len(pxcy_list))]))
		return ut.calcCondEnt(pzcy,py)+errzcyddot+0.5*pcoeff*errzcynorm2
	return funcObj
def gradPzcyObj(pxcy_list,py,pcoeff):
	def gradObj(pzcx_list,pzcy,muzcy_list):
		errzcy_list = np.array([pzcy-pzcx_list[i]@pxcy_list[i] for i in range(len(pxcy_list))])
		return -(np.log(pzcy)+1)*py[None,:]+np.sum(np.array(muzcy_list),0)+pcoeff*np.sum(errzcy_list,0)
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