import numpy as np
import sys
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


class singleViewTest:
	def __init__(self,enc,pxy,**kwargs):
		px = np.sum(pxy,axis=1)
		py = np.sum(pxy,axis=0)
		pz = enc@px
		self.pzcx = enc
		self.pxy = pxy
		self.pxcy = pxy/py[None,:]
		self.pycx = (pxy/px[:,None]).T
		self.pzcy = enc@self.pxcy
		self.pycz = ((self.pzcy*py[None,:])/pz[:,None]).T
		self.bayes_pycx = self.pycz @ self.pzcx
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))

	def test(self,x_test,y_test):
		# the x_test only contains single view
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)
		cumap = np.cumsum(self.bayes_pycx,axis=0)
		y_est  = np.zeros((num))
		for nn in range(num):
			yidx = -1
			for iy in range(self.pxy.shape[1]):
				if y_raw[nn]<cumap[iy,x_test[nn]]:
					yidx = iy
					break
			if yidx == -1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = yidx
		return y_est.astype(int)

class mvCcTest:
	def __init__(self,cons_enc_list,cmpl_enc,pxy_list,**kwargs):
		# focus on two view problems
		assert len(pxy_list) == 2
		view_idx = kwargs['view_idx']
		consensus_weight = kwargs['consensus_weight']
		self.pxy = pxy_list[view_idx]
		nx_list = [item.shape[0] for item in pxy_list]
		px = np.sum(self.pxy,axis=1)
		py = np.sum(self.pxy,axis=0)
		pxcy = self.pxy / py[None,:]
		self.py = py
		self.view_idx = view_idx
		self.cons_enc_list = cons_enc_list # this a consensus encoder (marginal)
		self.cmpl_enc = cmpl_enc # this a complement encoder (marginal)
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))
		#print(cons_enc_list[view_idx]) # say we use consensus encoder only, what's the accuracy///
		#sys.exit()
		this_con_enc = cons_enc_list[view_idx]
		self.this_con_enc = this_con_enc
		pxy = pxy_list[view_idx]
		pzy = this_con_enc@pxy
		pz = this_con_enc@px
		self.est_pycx = (pzy/pz[:,None]).T @ this_con_enc
		#self.this_con_enc = this_con_enc
		joint_pzcx = cmpl_enc * this_con_enc[None,...] # ze zc | x
		joint_pz = np.sum(joint_pzcx*px[...,:],axis=-1) # ze zc
		joint_pzzcy = joint_pzcx @pxcy # ze zc | y
		marg_py_zezc = (joint_pzzcy * py[...,:]) / joint_pz[...,None]  # NOTE: dim [ze,zc,y] but is y| ze,zc
		#marg_py_zezc/=np.sum(marg_py_zezc,axis=0)
		#print(np.sum(marg_py_zezc,axis=2))

		# design consensus map
		nz = this_con_enc.shape[0]
		assert np.all([item.shape[0]==nz for item in cons_enc_list])
		est_pyccx = np.zeros((py.shape[0],nz,nx_list[view_idx]))
		for iy in range(py.shape[0]):
			for ix in range(nx_list[view_idx]):
				for izc in range(nz):
					tmp_sum = 0
					for ize in range(this_con_enc.shape[0]):
						tmp_sum += marg_py_zezc[ize,izc,iy] * this_con_enc[izc,ix] * cmpl_enc[ize,izc,ix]
					est_pyccx[iy,izc,ix] = tmp_sum
		est_pyccx/=np.sum(est_pyccx,axis=0)
		self.est_pyccx = est_pyccx
		
		cons_map = np.zeros((nz,nz,nz))
		# must be equal sized
		for iz1 in range(nz):
			for iz2 in range(nz):
				if view_idx==0:
					if iz1==iz2:
						cons_map[iz1,iz1,iz2] = 1.0
						cons_map[(iz1+1)%2,iz1,iz2] = 0.0
					else:
						cons_map[iz1,iz1,iz2] = consensus_weight
						cons_map[(iz1+1)%2,iz1,iz2] = 1.0- consensus_weight
				else:
					if iz1==iz2:
						cons_map[iz2,iz1,iz2] = 1.0
						cons_map[(iz2+1)%2,iz1,iz2] = 0.0
					else:
						cons_map[iz2,iz1,iz2] = consensus_weight
						cons_map[(iz2+1)%2,iz1,iz2] = 1.0 - consensus_weight
		#print(cons_map)
		#print(np.sum(cons_map,axis=0))
		# zc consensus is an design parameter...
		est_pycx = np.zeros((py.shape[0],nx_list[0],nx_list[1]))#y, x1, x2
		for iy in range(py.shape[0]):
			for ix1 in range(nx_list[0]):
				for ix2 in range(nx_list[1]):
					tmp_sum =0
					for ize in range(cmpl_enc.shape[0]):# ne must match
						for izc in range(nz):
							inner_sum = 0
							for iz1 in range(nz):
								for iz2 in range(nz):
									inner_sum += cons_map[izc,iz1,iz2] * (cons_enc_list[0])[iz1,ix1] * (cons_enc_list[1])[iz2,ix2]
							if view_idx == 0:
								tmp_sum+= inner_sum * marg_py_zezc[ize,izc,iy] * cmpl_enc[ize,izc,ix1]
							else:
								tmp_sum+= inner_sum * marg_py_zezc[ize,izc,iy] * cmpl_enc[ize,izc,ix2]
					est_pycx[iy,ix1,ix2] = tmp_sum
		#print(np.sum(est_pycx,axis=0))
		est_pycx /= np.sum(est_pycx,axis=0)
		self.estz_pycx = est_pycx
		#sys.exit()
		
	def test_z(self,x_test,y_test):
		# x_test is one dimension
		num = x_test.shape[0]
		z_raw = self.rs.rand(num)
		y_raw = self.rs.rand(num)
		z_cumap = np.cumsum(self.this_con_enc,axis=0)
		#print(z_cumap)
		cumap = np.cumsum(self.est_pyccx,axis=0)
		#print(cumap)
		z_est = np.zeros(num,dtype=int)
		for nn in range(num):
			z_idx = -1
			for iz in range(self.this_con_enc.shape[0]):
				if z_raw[nn] < z_cumap[iz,x_test[nn]]:
					z_idx = iz
					break
			if z_idx==-1:
				print('ERROR:no encoding found')
				sys.exit()
			z_est[nn] = z_idx
		y_est = np.zeros(num,dtype=int)
		for nn in range(num):
			y_idx = -1
			for iy in range(self.py.shape[0]):
				if y_raw[nn]<cumap[iy,z_est[nn],x_test[nn]]:
					y_idx = iy
					break
			if y_idx ==-1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx
		return y_est.astype(int)
	def test_c(self,x_test,y_test):
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)
		cumap = np.cumsum(self.est_pycx,axis=0)
		y_est = np.zeros(num,dtype=int)
		for nn in range(num):
			y_idx = -1
			for iy in range(self.est_pycx.shape[0]):
				if y_raw[nn]<cumap[iy,x_test[nn]]:
					y_idx = iy
					break
			if y_idx == -1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx
		return y_est.astype(int)
	def test(self,x_test,y_test):
		# x_test is in [x0,x1] order
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)

		cumap = np.cumsum(self.estz_pycx,axis=0)
		y_est = np.zeros(num)

		for nn in range(num):
			y_idx = -1
			for iy in range(len(self.py)):
				ctuple = tuple([iy]+list(x_test[nn]))
				if y_raw[nn]<cumap[ctuple]:
					y_idx = iy
					break
			if y_idx==-1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx

		return y_est.astype(int)


class mvCcFullTest:
	def __init__(self,cons_enc_list,cmpl_enc_list,pxy_list,**kwargs):
		# focus on two view problems
		assert len(pxy_list) == 2
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))
		nx_list = [item.shape[0] for item in pxy_list]
		px_list = [np.sum(item,axis=1) for item in pxy_list]
		py = np.sum(pxy_list[0],axis=0)
		self.py = py
		pxcy_list = [item/py[None,:] for item in pxy_list]
		cons_enc_v1 = cons_enc_list[0]
		nz = cons_enc_v1.shape[0]
		cons_enc_v2 = cons_enc_list[1]
		assert nz == cons_enc_v2.shape[0]
		
		con_pzy = cons_enc_v1@pxy_list[0] # this should be the same for the two
		con_pzcy = cons_enc_v1@pxcy_list[0]
		con_pz = cons_enc_v1@px_list[0] # this is also the same for the two
		cmpl_enc_v1 = cmpl_enc_list[0]
		cmpl_enc_v2 = cmpl_enc_list[1]

		joint_pzx_v1 = cons_enc_v1 * (px_list[0])[None,:]
		joint_pzx_v2 = cons_enc_v2 * (px_list[1])[None,:]
		cmpl_dec_v1 = ((cmpl_enc_v1 * cons_enc_v1[None,...])@pxcy_list[0])/ con_pzcy[None,...]
		cmpl_dec_v2 = ((cmpl_enc_v2 * cons_enc_v2[None,...])@pxcy_list[1])/ con_pzcy[None,...]

		tmp_cmpl_dec = np.zeros((py.shape[0],cmpl_enc_v1.shape[0],cmpl_enc_v2.shape[0],nz,nz))
		for iy in range(py.shape[0]):
			for ize1 in range(cmpl_enc_v1.shape[0]):
				for ize2 in range(cmpl_enc_v2.shape[0]):
					for izc1 in range(nz):
						for izc2 in range(nz):
							tmp_cmpl_dec[iy,ize1,ize2,izc1,izc2] = py[iy]*con_pzcy[izc1,iy]*con_pzcy[izc2,iy]*cmpl_dec_v1[ize1,izc1,iy]*cmpl_dec_v2[ize2,izc2,iy]
		tmp_cmpl_dec/= np.sum(tmp_cmpl_dec,axis=0)
		# zc consensus is an design parameter...
		est_pycx = np.zeros((py.shape[0],nx_list[0],nx_list[1]))#y, x1, x2
		for iy in range(py.shape[0]):
			for ix1 in range(nx_list[0]):
				for ix2 in range(nx_list[1]):
					tmp_sum =0
					for ize1 in range(cmpl_enc_v1.shape[0]):# ne must match
						for ize2 in range(cmpl_enc_v2.shape[0]):
							for izc1 in range(cons_enc_v1.shape[0]):
								for izc2 in range(cons_enc_v2.shape[0]):
									tmp_sum += tmp_cmpl_dec[iy,ize1,ize2,izc1,izc2] * cmpl_enc_v1[ize1,izc1,ix1] * cons_enc_v1[izc1,ix1] * cmpl_enc_v2[ize2,izc2,ix2] * cons_enc_v2[izc2,ix2]
					est_pycx[iy,ix1,ix2] = tmp_sum
		est_pycx /= np.sum(est_pycx,axis=0)
		self.estz_pycx = est_pycx
		
	def test(self,x_test,y_test):
		# x_test is in [x0,x1] order
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)

		cumap = np.cumsum(self.estz_pycx,axis=0)
		y_est = np.zeros(num)

		for nn in range(num):
			y_idx = -1
			for iy in range(len(self.py)):
				ctuple = tuple([iy]+list(x_test[nn]))
				if y_raw[nn]<cumap[ctuple]:
					y_idx = iy
					break
			if y_idx==-1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx

		return y_est.astype(int)


class mvCcReviseTest:
	def __init__(self,cons_enc_list,cmpl_enc_list,pxy_list,**kwargs):
		# focus on two view problems
		assert len(pxy_list) == 2
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))
		nx_list = [item.shape[0] for item in pxy_list]
		px_list = [np.sum(item,axis=1) for item in pxy_list]
		py = np.sum(pxy_list[0],axis=0)
		self.py = py
		pxcy_list = [item/py[None,:] for item in pxy_list]
		# compute joint prior first
		prior_ycx1x2 = np.zeros((len(py),nx_list[0],nx_list[1]))
		prior_x1x2   = np.zeros((nx_list[0],nx_list[1]))
		for ix1 in range(nx_list[0]):
			for ix2 in range(nx_list[1]):
				tmp_sum = 0
				for iy in range(len(py)):
					tmp_sum += py[iy] * pxcy_list[0][ix1,iy] * pxcy_list[1][ix2,iy]
				prior_x1x2[ix1,ix2] = tmp_sum
		prior_x1x2 /= np.sum(prior_x1x2)
		# ycx1x2
		for ix1 in range(nx_list[0]):
			for ix2 in range(nx_list[1]):
				for iy in range(len(py)):
					prior_ycx1x2[iy,ix1,ix2] = py[iy] * pxcy_list[0][ix1,iy] * pxcy_list[1][ix2,iy] / prior_x1x2[ix1,ix2]
		prior_ycx1x2/= np.sum(prior_ycx1x2,axis=0)
		# 
		cons_enc_v1 = cons_enc_list[0]
		nz = cons_enc_v1.shape[0]
		cons_enc_v2 = cons_enc_list[1]
		assert nz == cons_enc_v2.shape[0]
		
		con_pzy = cons_enc_v1@pxy_list[0] # this should be the same for the two
		con_pzcy = cons_enc_v1@pxcy_list[0]
		con_pz = cons_enc_v1@px_list[0] # this is also the same for the two
		cmpl_enc_v1 = cmpl_enc_list[0]
		cmpl_enc_v2 = cmpl_enc_list[1]

		joint_pzx_v1 = cons_enc_v1 * (px_list[0])[None,:]
		joint_pzx_v2 = cons_enc_v2 * (px_list[1])[None,:]
		cmpl_dec_v1 = ((cmpl_enc_v1 * cons_enc_v1[None,...])@pxcy_list[0])/ con_pzcy[None,...]
		cmpl_dec_v2 = ((cmpl_enc_v2 * cons_enc_v2[None,...])@pxcy_list[1])/ con_pzcy[None,...]

		tmp_cmpl_dec = np.zeros((py.shape[0],cmpl_enc_v1.shape[0],cmpl_enc_v2.shape[0],nz)) # only one consensus
		for iy in range(py.shape[0]):
			for ize1 in range(cmpl_enc_v1.shape[0]):
				for ize2 in range(cmpl_enc_v2.shape[0]):
					for izc in range(nz):
						tmp_cmpl_dec[iy,ize1,ize2,izc] = py[iy] * con_pzcy[izc,iy] * cmpl_dec_v1[ize1,izc,iy]* cmpl_dec_v2[ize2,izc,iy]
		tmp_cmpl_dec/= np.sum(tmp_cmpl_dec,axis=0)
		# compute the joint encoder
		tmp_con_enc = np.zeros((nz,nx_list[0],nx_list[1]))
		'''
		for iz in range(nz):
			for ix1 in range(cmpl_enc_v1.shape[0]):
				for ix2 in range(cmpl_enc_v2.shape[1]):
					tmp_sum = 0
					for iy in range(len(py)):
						tmp_sum += prior_ycx1x2[iy,ix1,ix2] * con_pzcy[iz,iy]
					tmp_con_enc[iz,ix1,ix2] = tmp_sum
		tmp_con_enc/=np.sum(tmp_con_enc,axis=0)
		'''
		for iz in range(nz):
			for ix1 in range(nx_list[0]):
				for ix2 in range(nx_list[1]):
					tmp_norm = 0
					for sz in range(nz):
						tmp_norm += cons_enc_v1[sz,ix1] * cons_enc_v2[sz,ix2]
					tmp_con_enc[iz,ix1,ix2] = cons_enc_v1[sz,ix1] * cons_enc_v2[sz,ix2] / tmp_norm
		tmp_con_enc/=np.sum(tmp_con_enc,axis=0)
		
		# zc consensus is an design parameter...
		est_pycx = np.zeros((py.shape[0],nx_list[0],nx_list[1]))#y, x1, x2
		for iy in range(py.shape[0]):
			for ix1 in range(nx_list[0]):
				for ix2 in range(nx_list[1]):
					tmp_sum =0
					for ize1 in range(cmpl_enc_v1.shape[0]):# ne must match
						for ize2 in range(cmpl_enc_v2.shape[0]):
							for izc in range(cons_enc_v1.shape[0]):
								# should be forced to the same consensus
								tmp_sum += tmp_cmpl_dec[iy,ize1,ize2,izc] * cmpl_enc_v1[ize1,izc,ix1] * cmpl_enc_v2[ize2,izc,ix2] * tmp_con_enc[izc,ix1,ix2]
					est_pycx[iy,ix1,ix2] = tmp_sum
		est_pycx /= np.sum(est_pycx,axis=0)
		self.estz_pycx = est_pycx
		
	def test(self,x_test,y_test):
		# x_test is in [x0,x1] order
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)

		cumap = np.cumsum(self.estz_pycx,axis=0)
		y_est = np.zeros(num)

		for nn in range(num):
			y_idx = -1
			for iy in range(len(self.py)):
				ctuple = tuple([iy]+list(x_test[nn]))
				if y_raw[nn]<cumap[ctuple]:
					y_idx = iy
					break
			if y_idx==-1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx

		return y_est.astype(int)

'''
class mvIncTest:
	def __init__(self,enc_list,pxy_list,**kwargs):
		self.pxy_list = pxy_list
		
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))


	def test(self,x_test,y_test):
		return None
'''
class mvIncTestTwo:
	def __init__(self,enc_list,dec_list,pxy_list,**kwargs):
		# expect the encoders and decoders sorted with MI in descending order
		# as well as the pxy_list
		# enc_list: should be conditional marginal...
		assert len(enc_list) == 2
		# limited to two views for simplicity
		self.rs  = 	RandomState(MT19937(SeedSequence(kwargs['seed'])))
		self.pxy_list = pxy_list
		py = np.sum(pxy_list[0],axis=0)
		nx_list = [item.shape[0] for item in pxy_list]
		# expecting conditional joint backward decoder
		joint_dec = dec_list[-1] # z2z1|y   # correct
		joint_pz = np.sum(joint_dec * py[...,:],axis=-1) # z2 z1

		joint_pred = np.transpose(joint_dec * py[...,:] / joint_pz[...,None],(2,0,1) ) # y|z2 z1
		est_pycx = np.zeros((len(py),nx_list[0],nx_list[1])) # y|x0,x1
		
		for iy in range(len(py)):
			for ix1 in range(nx_list[0]):
				for ix2 in range(nx_list[1]):
					tmp_sum =0 
					# summing all representation
					for iz2 in range(enc_list[1].shape[0]):
						for iz1 in range(enc_list[0].shape[0]):
							tmp_sum += joint_pred[iy,iz2,iz1] * (enc_list[1])[iz2,iz1,ix2] * (enc_list[0])[iz1,ix1]
					est_pycx[iy,ix1,ix2] = tmp_sum
		self.est_pycx = est_pycx # the cumsum looks good
		self.py = py

	def test(self,x_test,y_text):
		# x_test is in [x0,x1] order
		num = x_test.shape[0]
		y_raw = self.rs.rand(num)

		cumap = np.cumsum(self.est_pycx,axis=0)
		y_est = np.zeros(num)

		for nn in range(num):
			y_idx = -1
			for iy in range(len(self.py)):
				ctuple = tuple([iy]+list(x_test[nn]))
				if y_raw[nn]<cumap[ctuple]:
					y_idx = iy
					break
			if y_idx==-1:
				print('ERROR:no prediction found')
				sys.exit()
			y_est[nn] = y_idx

		return y_est.astype(int)
