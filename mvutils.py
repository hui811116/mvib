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
	val_mask = pzezcx>1e-9
	pzezc = np.sum(pzezcx,axis=-1)
	pzcx = np.sum(pzezcx,axis=0)
	pzc = np.sum(pzezcx,axis=(0,2))
	px = np.sum(pzezcx,axis=(0,1))
	pzeczc = pzezc/pzc[None,:]
	pzeczcx = pzezcx/px[...,:]/pzeczc[...,None]
	expand_pzeczc = np.repeat(np.expand_dims(pzeczc,axis=-1),len(px),axis=-1)
	return np.sum(pzezcx[val_mask]*np.log(pzeczcx/expand_pzeczc)[val_mask])

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
