import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import sys
import mvgrad as gd
import copy

def tf_ib_orig(tf_pxy,gamma,convthres,maxiter,**kwargs):
	(nx,ny) = tf_pxy.shape
	tf_py = tf.reduce_sum(tf_pxy,axis=0)
	tf_px = tf.reduce_sum(tf_pxy,axis=1)
	nz = ny
	pycx = tf.transpose(tf_pxy/tf_px[:,None])
	pxcy = tf_pxy/tf_py[None,:]
	tf_pzcx = tf.nn.softmax(tf.keras.backend.random_normal(shape=(nz,nx)),axis=0)
	tf_pz = tf.squeeze(tf_pzcx @ tf_px[:,None])
	tf_pycz = tf.transpose(tf_pzcx @ tf_pxy /tf_pz[:,None])

	#
	itcnt = 0
	conv_flag = False
	while itcnt< maxiter:
		itcnt +=1
		kl_oprod = tf.expand_dims(1/tf_pycz,axis=-1)@tf.expand_dims(pycx,axis=1)
		kl_ker = tf.reduce_sum(tf.repeat(tf.expand_dims(pycx,axis=1),nz,axis=1)*tf.math.log(kl_oprod),axis=0)
		new_pzcx = tf.linalg.diag(tf_pz)@tf.math.exp(-1/gamma* kl_ker)
		new_pzcx /= tf.reduce_sum(new_pzcx,axis=0)
		diff= 0.5* tf.reduce_sum(tf.math.abs(new_pzcx - tf_pzcx))
		if diff < convthres:
			conv_flag = True
			break
		else:
			tf_pzcx = new_pzcx
			tf_pz = tf.squeeze(tf_pzcx @ tf_px[:,None])
			tf_pycz = tf.transpose( (tf_pzcx @ tf_pxy)/tf_pz[:,None] )
	mixz = tf.reduce_sum(tf_pzcx* tf_px[None,:] * tf.math.log(tf_pzcx/tf_pz[:,None]))
	miyz = tf.reduce_sum(tf_pycz* tf_py[:,None] * tf.math.log(tf_pycz/tf_py[:,None]))
	return {'pzcx':tf_pzcx,'pycz':tf_pycz,'niter':itcnt,
			'IXZ':mixz.numpy(),'IYZ':miyz.numpy(),'conv':conv_flag}


def tf_admmib_type1(tf_pxy,gamma,convthres,maxiter,**kwargs):
	ss_scale = kwargs['ss_scale']
	ss_init = kwargs['ss_init']
	penalty = kwargs['penalty']
	# first, using numpy to prepare the vectors and tensors
	(nx,ny) = tf_pxy.shape
	nz= ny
	tf_px = tf.reduce_sum(tf_pxy,axis=1)
	tf_py = tf.reduce_sum(tf_pxy,axis=0)
	tf_pycx = tf.transpose(tf_pxy/tf_px[:,None])
	tf_pxcy = tf_pxy/ tf_py[None,:]
	# random initialization
	tf_pzcx = tf.nn.softmax(tf.random.normal(shape=(nz,nx)),axis=0)
	tf_pz = tf.squeeze(tf_pzcx@tf_px[:,None])
	# normalization to be a valid condition prob
	dual_z  = tf.zeros((nz))
	itcnt = 0
	conv= False

	while itcnt < maxiter:
		itcnt += 1
		# strongly convex update
		err_z = tf_pz - tf.squeeze(tf_pzcx@tf_px[:,None])
		grad_pz = (1-gamma)*(tf.math.log(tf_pz)+1)+dual_z+penalty*err_z
		mean_grad_z = grad_pz - tf.reduce_mean(grad_pz)
		ss_z=  gd.tfNaiveSS(tf_pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		new_pz = tf_pz - mean_grad_z*ss_z
		# the stepsize selection is divided into two step,
		err_z = new_pz - tf.squeeze(tf_pzcx@tf_px[:,None])
		dual_z += penalty * err_z
		tf_pzcy = tf_pzcx @ tf_pxcy
		grad_x = (gamma*(tf.math.log(tf_pzcx)+1) - (tf.math.log(tf_pzcy)+1)@tf_pycx - (dual_z+penalty*err_z)[:,None])*tf_px[None,:]
		mean_grad_x = grad_x - tf.reduce_mean(grad_x,axis=0)
		ss_x = gd.tfNaiveSS(tf_pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		new_pzcx = tf_pzcx - ss_x * mean_grad_x
		err_z = new_pz - tf.squeeze(new_pzcx@tf_px[:,None])
		dtv_z = 0.5 * tf.reduce_sum(tf.math.abs(err_z))
		if dtv_z < convthres:
			conv=True
			break
		else:
			tf_pz = new_pz
			tf_pzcx = new_pzcx
	# calculate mutual information
	mizx = tf.reduce_sum(tf_pzcx*tf_px[None,:]*tf.math.log(tf_pzcx/tf_pz[:,None]))
	tf_pzcy = tf_pzcx @ tf_pxcy
	mizy = tf.reduce_sum(tf_pzcy*tf_py[None,:]*tf.math.log(tf_pzcy/tf_pz[:,None]))
	return {'pzcx':tf_pzcx,'niter':itcnt,'conv':conv,'IZX':mizx.numpy(),'IZY':mizy.numpy()}

# simply copy, need debugging
def tf_mvcc_nv(tf_pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	nview = len(tf_pxy_list)
	# preparing the priors
	tf_py = tf.reduce_sum(tf_pxy_list[0],axis=0)
	ny = len(tf_py)
	#
	tf_px_list  = [tf.reduce_sum(item,axis=1) for item in tf_pxy_list]
	tf_pxcy_list = [item/tf_py[None,:] for item in tf_pxy_list]
	#nz = np.amin([item.shape[0] for item in tf_pxy_list])
	nz = ny
	# initialization
	tf_pzcx_list = [tf.nn.softmax(tf.keras.backend.random_normal(shape=(nz,item.shape[0]) ), axis=0) for item in tf_pxy_list]
	tf_pz= 1/nview * sum([tf.reduce_sum(tf_pzcx_list[i]*tf_px_list[i][None,:],axis=1) for i in range(nview)])
	tf_pz /= tf.reduce_sum(tf_pz)
	tf_pzcy = 1/nview*sum([tf_pzcx_list[i]@tf_pxcy_list[i] for i in range(nview)])
	# dual variables
	dz_list = [tf.zeros(nz) for i in range(nview)]
	dzcy_list = [tf.zeros((nz,ny)) for i in range(nview)]
	# counter and flags
	itcnt = 0
	flag_conv = False
	# local functions

	def tfGradPzcx(gamma,tfpx,tfpxcy,penalty,tfpzcx,tfpz,tfpzcy,tfdualz,tfdualzcy):
		err = tf.reduce_sum(tfpzcx*tfpx[None,:],axis=1)-tfpz
		errzy = tfpzcx@tfpxcy - tfpzcy
		return gamma * (tf.math.log(tfpzcx)+1)*tfpx[None,:]+(tfdualz+penalty * err)[:,None]*tfpx[None,:]\
				+(tfdualzcy+penalty*errzy)@tf.transpose(tfpxcy)
	def tfGradPz(gamma_vec,tfpx_list,penalty,tfpz,tfpzcx_list,dzlist):
		errz_list = [tf.reduce_sum(tfpzcx_list[ii]*tfpx_list[ii][None,:],axis=1)-tfpz for ii in range(nview)]
		return (1-np.sum(gamma_vec))*(tf.math.log(tfpz)+1)-sum(dzlist)-penalty*sum(errz_list)
	def tfGradPzcy(tfpxcy_list,tfpy,penalty,tfpzcy,tfpzcx_list,tfdualzcy_list):
		errzcy_list = [tfpzcx_list[ii]@tfpxcy_list[ii]-tfpzcy for ii in range(nview)]
		return -(tf.math.log(tfpzcy)+1)*tfpy[None,:]-sum(tfdualzcy_list)-penalty*sum(errzcy_list)
	# step size control

	while itcnt< maxiter:
		itcnt+=1
		# update the primal variables
		_ss_interrupt = False
		new_pzcx_list = [tf.zeros((nz,item.shape[0])) for item in tf_pxy_list]
		for ii in range(nview):
			# prepare the error 
			tmp_grad_pzcx = tfGradPzcx(gamma_vec[ii],tf_px_list[ii],tf_pxcy_list[ii],penalty,
										tf_pzcx_list[ii],tf_pz,tf_pzcy,dz_list[ii],dzcy_list[ii])
			mean_tmp_grad_pzcx = tmp_grad_pzcx - tf.reduce_mean(tmp_grad_pzcx,axis=0)[None,:]
			ss_x = gd.tfNaiveSS(tf_pzcx_list[ii],-mean_tmp_grad_pzcx,ss_init,ss_scale)
			if ss_x == 0:
				#print('debugging: tfmvcc xstep failed. iter:{}'.format(itcnt))
				_ss_interrupt=True
				break
			new_pzcx_list[ii] = tf_pzcx_list[ii] - ss_x*mean_tmp_grad_pzcx
		if _ss_interrupt:
			break
		# update the augmented variable
		grad_pz = tfGradPz(gamma_vec,tf_px_list,penalty,tf_pz,new_pzcx_list,dz_list)
		mean_grad_pz = grad_pz - tf.reduce_mean(grad_pz)
		ss_z = gd.tfNaiveSS(tf_pz,-mean_grad_pz,ss_init,ss_scale)
		if ss_z == 0:
			#print('debugging: tfmvcc zstep failed')
			break
		grad_pzcy = tfGradPzcy(tf_pxcy_list,tf_py,penalty,tf_pzcy,new_pzcx_list,dzcy_list)
		mean_grad_pzcy = grad_pzcy - tf.reduce_mean(grad_pzcy,axis=0)[None,:]
		ss_y = gd.tfNaiveSS(tf_pzcy,-mean_grad_pzcy,ss_z,ss_scale)
		if ss_y == 0:
			#print('debugging: tfmvcc ystep failed')
			break
		new_tfpz = tf_pz - ss_y * mean_grad_pz
		new_tfpzcy = tf_pzcy - ss_y * mean_grad_pzcy
		# dual updates
		errz_list = [tf.reduce_sum(item*tf_px_list[idx][None,:],axis=1)-new_tfpz for idx,item in enumerate(new_pzcx_list)]
		errzcy_list = [item@tf_pxcy_list[idx]-new_tfpzcy for idx, item in enumerate(new_pzcx_list)]
		dz_list = [item + penalty * errz_list[idx] for idx, item in enumerate(dz_list)]
		dzcy_list = [item + penalty * errzcy_list[idx] for idx, item in enumerate(dzcy_list)]
		# convergence criterion

		conv_z_list = tf.convert_to_tensor([0.5* tf.reduce_sum(tf.math.abs(item)) for item in errz_list],dtype=tf.float32)
		conv_zcy_list = tf.convert_to_tensor([0.5*tf.reduce_sum(tf.math.abs(item),axis=0) for item in errzcy_list],dtype=tf.float32)
		if tf.math.reduce_all(conv_z_list<convthres) and tf.math.reduce_all(conv_zcy_list<convthres):
			flag_conv = True
			break
		else:
			tf_pzcx_list = new_pzcx_list
			tf_pz = new_tfpz
			tf_pzcy = new_tfpzcy
	mizx_list = [tf.reduce_sum(tf_pzcx_list[idx]*tf_px_list[idx][None,:]*tf.math.log(tf_pzcx_list[idx]/tf_pz[:,None])).numpy() for idx in range(nview)]
	mizy_list = [tf.reduce_sum(tf_pzcx_list[idx]@tf_pxy_list[idx]*tf.math.log((tf_pzcx_list[idx]@tf_pxcy_list[idx])/tf_pz[:,None])).numpy() for idx in range(nview)]
	return {'pzcx_list':tf_pzcx_list,'pz':tf_pz,'pzcy':tf_pzcy,'niter':itcnt,'conv':flag_conv,
			'IXZ_list':mizx_list,'IYZ_list':mizy_list}

# simply copy need debugging
def tf_mvcc_cmpl_type2(tf_pxy,tf_enc_pzcx,gamma,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	(nx,ny) = tf_pxy.shape
	nc = tf_enc_pzcx.shape[0]
	tf_px = tf.reduce_sum(tf_pxy,axis=1)
	ne = tf_px.shape[0]
	#ne = ny
	tf_py = tf.reduce_sum(tf_pxy,axis=0)
	tf_pxcy = tf_pxy / tf_py[None,:]
	tf_pycx = tf.transpose(tf_pxy/tf_px[:,None])
	# some constants
	tf_const_pzx = tf_enc_pzcx * tf_px[None,:]
	tf_const_pz = tf.squeeze(tf_enc_pzcx@ tf_px[:,None])
	tf_const_pzy = tf_enc_pzcx@tf_pxy
	tf_const_pzcy = tf_enc_pzcx@tf_pxcy
	tf_const_grad_x = tf_const_pzx / tf_const_pz[:,None]

	tf_var_pzcx = tf.nn.softmax(tf.keras.backend.random_normal(shape=(ne,nc,nx)),axis=0)

	tf_var_pz = tf.reduce_sum(tf_var_pzcx*tf_enc_pzcx[None,...] * tf_px[...,:],axis=-1)/ tf_const_pz[None,:]
	tf_var_pzcy = ((tf_var_pzcx * tf_enc_pzcx[None,...])@tf_pxcy)/((tf_enc_pzcx@tf_pxcy)[None,...])

	# dual variables
	dual_z = tf.zeros((ne,nc))
	dual_zy = tf.zeros((ne,nc,ny))

	itcnt =0
	conv_flag = False
	while itcnt<maxiter:
		itcnt+=1
		errz = tf.reduce_sum(tf_var_pzcx*tf_const_pzx[None,...],axis=-1)/(tf_const_pz[None,:]) - tf_var_pz
		errzy = ((tf_var_pzcx * tf_enc_pzcx[None,...])@tf_pxcy)/((tf_const_pzcy)[None,...]) - tf_var_pzcy
		grad_x = gamma*(tf.math.log(tf_var_pzcx)+1)*tf_const_pzx[None,...]+(dual_z+penalty*errz)[...,None]*tf_const_grad_x[None,...]\
				+((dual_zy+penalty*errzy)/tf_const_pzcy[None,...])@tf.transpose(tf_pxcy)*tf_enc_pzcx[None,...]
		mean_grad_x = grad_x - tf.reduce_mean(grad_x,axis=0)
		x_ss = gd.tfNaiveSS(tf_var_pzcx,-mean_grad_x,ss_init,ss_scale)
		if x_ss == 0:
			break
		new_tf_var_pzcx = tf_var_pzcx -x_ss * mean_grad_x
		# H(Z_e|Z_c)
		errz = tf.reduce_sum(new_tf_var_pzcx*tf_const_pzx[None,...],axis=-1)/(tf_const_pz[None,:]) - tf_var_pz
		errzy = ((new_tf_var_pzcx*tf_enc_pzcx[None,...])@tf_pxcy)/((tf_const_pzcy)[None,...]) - tf_var_pzcy

		grad_z = (1-gamma)*(tf.math.log(tf_var_pz)+1)*tf_const_pz[None,:]-(dual_z+penalty*errz)
		mean_grad_z=  grad_z - tf.reduce_mean(grad_z,axis=0)
		z_ss = gd.tfNaiveSS(tf_var_pz,-mean_grad_z,ss_init,ss_scale)
		if z_ss == 0:
			break
		grad_y = -(tf.math.log(tf_var_pzcy))*tf_const_pzy[None,...] - (dual_zy+penalty*errzy)
		mean_grad_y = grad_y - tf.reduce_mean(grad_y,axis=0)
		y_ss = gd.tfNaiveSS(tf_var_pzcy,-mean_grad_y,z_ss,ss_scale)
		if y_ss ==0:
			break
		new_tf_var_pz = tf_var_pz - mean_grad_z*z_ss
		new_tf_var_pzcy = tf_var_pzcy -mean_grad_y * y_ss
		errz = tf.reduce_sum(new_tf_var_pzcx*tf_const_pzx[None,...],axis=-1)/(tf_const_pz[None,...]) - new_tf_var_pz
		errzy = ((new_tf_var_pzcx * tf_enc_pzcx[None,...])@tf_pxcy)/((tf_const_pzcy)[None,...]) - new_tf_var_pzcy

		dual_z += penalty * errz
		dual_zy += penalty * errzy
		dtvz = 0.5 * tf.reduce_sum(tf.math.abs(errz),axis=0)
		dtvzy = 0.5* tf.reduce_sum(tf.math.abs(errzy),axis=0)
		if tf.reduce_all(dtvz<convthres) and tf.reduce_all(dtvzy<convthres):
			conv_flag = True
			break
		else:
			tf_var_pzcx = new_tf_var_pzcx
			tf_var_pz = new_tf_var_pz
			tf_var_pzcy = new_tf_var_pzcy
	joint_pzcx = tf_var_pzcx * tf_const_pzx[None,...]
	joint_pzcy = tf_var_pzcy * tf_const_pzy[None,...]
	miczx = tf.reduce_sum(joint_pzcx * tf.math.log(tf_var_pzcx/tf_var_pz[...,None]))
	miczy = tf.reduce_sum(joint_pzcy * tf.math.log(tf_var_pzcy/tf_var_pz[...,None]))
	return {'pzeccx':tf_var_pzcx,'niter':itcnt,'conv':conv_flag,
			'IZCX':miczx.numpy(),'IZCY':miczy.numpy()}


def tf_inc_single_type2(tf_pxy,gamma,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	tf_py = tf.reduce_sum(tf_pxy,axis=0)
	tf_px = tf.reduce_sum(tf_pxy,axis=1)
	#nz = tf_px.shape[0]
	nz = len(tf_py)
	tf_pxcy = tf_pxy/ tf_py[None,:]
	tf_pycx = tf.transpose(tf_pxy /tf_px[:,None])
	(nx,ny) = tf_pxy.shape
	tf_prior_pzcy = kwargs['prior_pzcy']
	tf_prior_pz = tf.reduce_sum(tf_prior_pzcy*tf_py[...,:],axis=-1)

	tf_prior_pzcx = tf_prior_pzcy @ tf_pycx
	tf_const_grad_x_scalar = tf_prior_pzcx * tf_px[...,:]
	tf_const_grad_y_scalar = tf_prior_pzcy * tf_py[...,:]
	tf_const_gradx_pendz = (tf_const_grad_x_scalar)/tf_prior_pz[...,None]

	tmp_pzcx_shape = tuple([nz]+list(tf_prior_pzcx.shape))
	tf_var_pzcx = tf.nn.softmax(tf.keras.backend.random_normal(shape=tmp_pzcx_shape),axis=0)
	tf_var_pz = tf.reduce_sum( (tf_prior_pzcx*tf_px[...,:])[None,...]*tf_var_pzcx,axis=-1)/tf_prior_pz[None,...]
	tf_var_pzcy = tf_var_pzcx @ tf_pxcy

	dual_z = tf.zeros(tf_var_pz.shape)
	dual_zy = tf.zeros(tf_var_pzcy.shape)

	# system
	itcnt = 0
	conv_flag = False
	while itcnt < maxiter:
		itcnt+=1

		errz = tf.reduce_sum((tf_const_grad_x_scalar)[None,:]*tf_var_pzcx,axis=-1)/tf_prior_pz[None,...]-tf_var_pz
		errzy = tf_var_pzcx@tf_pxcy - tf_var_pzcy

		# -gamma H(Z|Z',X)
		grad_x = gamma*(tf.math.log(tf_var_pzcx)+1)*tf_const_grad_x_scalar[None,:] + (dual_z+penalty*errz)[...,None] * tf_const_gradx_pendz\
				+(dual_zy + penalty*errzy)@tf.transpose(tf_pxcy)
		mean_grad_x = grad_x - tf.reduce_mean(grad_x,axis=0)
		ss_x=  gd.tfNaiveSS(tf_var_pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		new_var_pzcx = tf_var_pzcx - mean_grad_x * ss_x

		errz = tf.reduce_sum((tf_const_grad_x_scalar)[None,:]*new_var_pzcx,axis=-1)/tf_prior_pz[None,...] - tf_var_pz
		errzy = new_var_pzcx@tf_pxcy - tf_var_pzcy

		# gradient of z
		grad_z = (1-gamma)*(tf.math.log(tf_var_pz)+1) * tf_prior_pz[None,...] - (dual_z + penalty*errz)
		mean_grad_z = grad_z - tf.reduce_mean(grad_z,axis=0)
		ss_z = gd.tfNaiveSS(tf_var_pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		# gradient of y
		grad_y = -(tf.math.log(tf_var_pzcy)+1) * tf_const_grad_y_scalar[None,...] - (dual_zy+penalty*errzy)
		mean_grad_y = grad_y - tf.reduce_mean(grad_y,axis=0)
		ss_y = gd.tfNaiveSS(tf_var_pzcy,-mean_grad_y,ss_z,ss_scale)
		if ss_y == 0:
			break
		new_var_pz = tf_var_pz - mean_grad_z * ss_y
		new_var_pzcy = tf_var_pzcy - mean_grad_y * ss_y

		errz = tf.reduce_sum((tf_const_grad_x_scalar)[None,...]*new_var_pzcx,axis=-1)/tf_prior_pz[None,...] - new_var_pz
		errzy = new_var_pzcx@tf_pxcy - new_var_pzcy

		dual_z +=  penalty * errz
		dual_zy +=  penalty * errzy

		# convergence check
		convz = 0.5* tf.reduce_sum(tf.math.abs(errz),axis=0)
		convzy = 0.5* tf.reduce_sum(tf.math.abs(errzy),axis=0)
		if tf.reduce_all(convz<convthres) and tf.reduce_all(convzy<convthres):
			conv_flag = True
			break
		else:
			tf_var_pzcx = new_var_pzcx
			tf_var_pz = new_var_pz
			tf_var_pzcy = new_var_pzcy
	out_backward_enc = tf_var_pzcy * tf_prior_pzcy[None,...]
	joint_pzcx = tf_var_pzcx * tf_const_grad_x_scalar
	mic_zcx = tf.reduce_sum(joint_pzcx*tf.math.log(tf_var_pzcx/tf_var_pz[...,None]))
	joint_pzcy = tf_var_pzcy * tf_const_grad_y_scalar
	mic_zcy = tf.reduce_sum(joint_pzcy*tf.math.log(tf_var_pzcy/tf_var_pz[...,None]))
	return {'pzczx':tf_var_pzcx,'pzzcy':out_backward_enc,'niter':itcnt,'conv':conv_flag,
			'IZCX':mic_zcx.numpy(),'IZCY':mic_zcy.numpy()}


# mvib mvcc_tf
def tf_mvib_cc(tf_pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	d_retry = kwargs['retry']
	nview = len(tf_pxy_list)
	tf_py = tf.reduce_sum(tf_pxy_list[0],axis=0)
	ny = len(tf_py)

	tf_px_list = [tf.reduce_sum(item,axis=1) for item in tf_pxy_list]
	tf_pxcy_list = [item/tf_py[None,:] for item in tf_pxy_list]

	# output of the consensus step
	outdict = tf_mvcc_nv(tf_pxy_list,gamma_vec,convthres,maxiter,**kwargs)
	if not outdict['conv']:
		print('ERROR:consensus failed')
		return {'conv':False}
	# debugging, print the learned MI
	print('LOG:consensus step converged')
	print(outdict['niter'],outdict['IXZ_list'],outdict['IYZ_list'])

	tmp_cmpl_list = []
	cmpl_izcx_list = []
	cmpl_izcy_list = []
	for vidx in range(nview):
		inner_loop_conv = False
		best_mic = -1.0
		best_out = {}
		for rn in range(d_retry):
			cmpl_out = tf_mvcc_cmpl_type2(tf_pxy_list[vidx],outdict['pzcx_list'][vidx],gamma_vec[vidx],convthres,maxiter,**kwargs)
			if not cmpl_out['conv']:
				print('ERROR: view {:>5} failed (retry count:{:>3})'.format(vidx,rn))
			else:
				if cmpl_out['IZCY']>best_mic:
					best_mic = cmpl_out['IZCY']
					best_out = copy.deepcopy(cmpl_out)
				inner_loop_conv = True
				print('LOG: view {:>5} converged (retry count:{:>3}): best_IYZC={:>10.5f}'.format(vidx,rn,best_mic))
		if not inner_loop_conv:
			return {'conv':False}
		else:
			print('LOG:complement view {:>5} converged: IXZC={:>10.4f}, IYZC={:>10.4f}'.format(vidx,best_out['IZCX'],best_out['IZCY']))
			tmp_cmpl_list.append(best_out['pzeccx'])
			cmpl_izcx_list.append(best_out['IZCX'])
			cmpl_izcy_list.append(best_out['IZCY'])
	print('LOG:convergence of mvib_cc reached!')
	return {'con_enc':outdict['pzcx_list'],'cmpl_enc':tmp_cmpl_list,
		'IXZ_list':outdict['IXZ_list'],'IYZ_list':outdict['IYZ_list'],
		'IXZC_list':cmpl_izcx_list,'IYZC_list':cmpl_izcy_list,'conv':True,
	}

def tf_mvib_inc(pxy_list,gamma_vec,convthres,maxiter,**kwargs):
	d_retry = kwargs['retry']
	nview = len(pxy_list)
	py = np.sum(pxy_list[0],axis=0)
	tf_py = tf.convert_to_tensor(py,dtype=tf.float32)
	ny = len(py)

	tf_pxy_list = [tf.convert_to_tensor(item,dtype=tf.float32) for item in pxy_list]
	tf_pxcy_list = [item/tf_py[None,:] for item in tf_pxy_list]
	tf_px_list = [item/tf_py[None,:] for item in tf_pxy_list]

	pzcy_prior = None
	tmp_est_list = []
	tmp_dec_list = []
	mizx_list = []
	mizy_list = []
	for vidx in range(nview):
		if vidx == 0:
			est_init = tf_ib_orig(tf_pxy_list[vidx],gamma_vec[vidx],convthres,maxiter,**kwargs)
			if est_init['conv']:
				print('LOG:incremental view {:>5} converged (BA)--IXZ={:>10.4f}, IYZ={:>10.4f}'.format(vidx,est_init['IXZ'],est_init['IYZ']))
				pzcy_prior = est_init['pzcx'] @ tf_pxcy_list[vidx]
				tmp_dec_list.append(pzcy_prior)
				tmp_est_list.append(est_init['pzcx'])
				mizx_list.append(est_init['IXZ'])
				mizy_list.append(est_init['IYZ'])
			else:
				return {'conv':False}
		else:
			inner_loop_conv = False
			best_out = {}
			best_mic = 0.0
			for rn in range(d_retry):
				est_out = tf_inc_single_type2(tf_pxy_list[vidx],gamma_vec[vidx],convthres,maxiter,**{'prior_pzcy':pzcy_prior,**kwargs})
				if not est_out['conv']:
					print('Error: view {:>5} failed (retry count:{:>3})'.format(vidx,rn))
				else:
					if est_out['IZCY']>best_mic:
						best_out = copy.deepcopy(est_out)
						best_mic = est_out['IZCY']
					inner_loop_conv=True
					print('LOG: view {:>5} converged (retry count:{:>3}): best_IYZC={:>10.5f}'.format(vidx,rn,best_mic))
			if not inner_loop_conv:
				return {'conv':False}
			else:
				print("LOG:incremental view {:>5} converged -- IXZ|Z'={:>10.4f}, IYZ|Z'={:>10.4f}".format(vidx,best_out['IZCX'],best_out['IZCY']))
				pzcy_prior = best_out['pzzcy']
				tmp_est_list.append(best_out['pzczx'])
				tmp_dec_list.append(best_out['pzzcy'])
				mizx_list.append(best_out['IZCX'])
				mizy_list.append(best_out['IZCY'])
	print('LOG:incremental method converged!')
	return {'conv':True,'enc_list':tmp_est_list,'dec_list':tmp_dec_list,
			'IXZ_list':mizx_list,'IYZ_list':mizy_list}
