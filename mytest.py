import numpy as np
import sys
import os


class singleView:
	def __init__(self,enc,pxy):
		self.pzcx = enc
		self.pxy = pxy
		self.pxcy = pxy/np.sum(pxy,axis=0)[None,:]
		self.px = np.sum(pxy,axis=1)
		self.py = np.sum(pxy,axis=0)
		self.pycx = (pxy/np.sum(pxy,axis=1)[:,None]).T
		self.pzcy = enc@self.pxcy
		self.pycz = ((self.pzcy@self.py)/(enc@self.px)[:,None]).T
	def test(self,x_test,y_test):
		pass
		return None
