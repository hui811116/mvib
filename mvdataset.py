import numpy as np
import sys

def select(name):
	if name == 'syn2':
		return synTwoView()
	else:
		sys.exit('[DATASET ERROR] No dataset named "{}" found.'.format(name))

def synTwoView():
	py = np.array([0.5,0.5])
	pxcy_v1 = np.array([[0.75,0],
						[0.25,0.25],
						[0,0.75]])
	pxcy_v2 = np.array([[0.85,0.15],
						[0.15,0.85]])
	pxy_v1 = pxcy_v1 * py[None,:]
	pxy_v2 = pxcy_v2 * py[None,:]
	return {'pxy_list':[pxy_v1,pxy_v2],'ny':len(py),'nz':2}