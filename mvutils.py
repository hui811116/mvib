import numpy as np
import sys

def calcMI(pxy):
	(nx,ny) = pxy.shape
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	msum = 0
	for i in range(nx):
		for j in range(ny):
			if pxy[i,j] !=0.0:
				msum += pxy[i,j] * np.log(pxy[i,j]/px[i]/py[j])
	return msum
	#return np.sum(pxy*np.log(1/px[:,None]*pxy/py[None,:]))
def calcEnt(px):
	esum = 0
	for i in range(len(px)):
		if px[i] > 0:
			esum -= px[i] * np.log(px[i])
	return esum

def calcMICmpl(pzezcx):
	# this is joint
	val_mask = pzezcx>1e-9
	(ne,nc,nx) = pzezcx.shape
	pzec = np.sum(pzezcx,axis=2)
	px = np.sum(pzezcx,axis=(0,1))
	pzc = np.sum(pzezcx,axis=(0,2))
	pzcx = np.sum(pzezcx,axis=0)/px[None,:]
	pzeccx = pzezcx/px[...,:]
	pzeczc = pzec/pzc[None,:]
	rep_pzcx = np.repeat(pzcx[None,...],ne,axis=0)
	rep_pzec = np.repeat(pzeczc[...,None],nx,axis=2)
	cond_ze_zcx = np.zeros(pzezcx.shape)
	cond_ze_zcx[val_mask] = pzeccx[val_mask]/rep_pzcx[val_mask]
	return np.sum(pzezcx[val_mask]*np.log(cond_ze_zcx[val_mask]/rep_pzec[val_mask]))

def calcCondEnt(pxcy,py):
	(nx,ny) = pxcy.shape
	assert ny == py.shape[0] #'the dimension of py and pxcy must be equal'
	pxy=  pxcy * py[None,:]
	esum = 0
	for i in range(nx):
		for j in range(ny):
			if pxy[i,j] > 0:
				esum -= pxy[i,j] * np.log(pxcy[i,j])
	return esum
