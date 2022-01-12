import numpy as np
import sys

def select(name):
	if name == 'syn2':
		return synTwoView234()
	elif name == 'syn2_simple':
		return synTwoView()
	elif name == 'syn1':
		return synOneView()
	elif name == 'syn2_inc':
		return synTwoViewInc()
	else:
		sys.exit('[DATASET ERROR] No dataset named "{}" found.'.format(name))

def getAvailableDataset():
	return ['syn2','syn2_simple','syn1','syn2_inc']

def synTwoView():
	py = np.array([0.5,0.5])
	# BSC, BEC views
	pxcy_v1 = np.array([[0.75,0.05],
						[0.20,0.20],
						[0.05,0.75]])
	pxcy_v2 = np.array([[0.85,0.15],
						[0.15,0.85]])
	pxy_v1 = pxcy_v1 * py[None,:]
	pxy_v2 = pxcy_v2 * py[None,:]
	return {'pxy_list':[pxy_v1,pxy_v2],'ny':len(py),'py':py}

def synTwoView234():
	py = np.array([0.5,0.5])
	# y,nx1,nx2
	pxcy_v1 = np.array([[0.75,0.05],
		                [0.20,0.20],
		                [0.05,0.75]])
	pxcy_v2 = np.array([[0.650, 0.025],
		                [0.250, 0.075],
		                [0.075, 0.250],
		                [0.025, 0.650]])
	# Trial and error, gamma_2 = 0.25 works
	pxy_v1 = pxcy_v1 * py[None,:]
	pxy_v2 = pxcy_v2 * py[None,:]
	return {'pxy_list':[pxy_v1,pxy_v2],'ny':len(py),'py':py}

def synTwoViewInc():
	py = np.array([1/3, 1/3, 1/3])
	py/= np.sum(py)
	pxcy_v1 = np.array([[0.90 ,0.20, 0.20],
						[0.05 ,0.45, 0.35],
						[0.05 ,0.35, 0.45]]) # clear for the first
	pxcy_v2 = np.array([[0.25 ,0.10, 0.55],
						[0.20 ,0.80, 0.25],
						[0.55 ,0.10, 0.20]])# clear for the second
	pxy_v1 = pxcy_v1 * py[None,:]
	pxy_v2 = pxcy_v2 * py[None,:]
	return {'pxy_list':[pxy_v1,pxy_v2],'ny':len(py),'py':py}
def synOneView():
	py = np.array([0.5,0.5])
	pxcy = np.array([[0.85, 0.15],
					 [0.15, 0.85]])
	pxy  = pxcy * py[None,:]
	return {'pxy_list':[pxy],'ny':len(py),'py':py}