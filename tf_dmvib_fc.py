import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import sys


class Sampling(layers.Layer):
	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
		return z_mean + tf.exp(0.5*z_log_var) * epsilon

latent_dim = 2
latent_dim_v2=3
shared_latent_dim = 2

nclass = 2
gamma_v1 = 1.0
gamma_v2 = 1.0

#encoder_inputs = keras.Input(shape=(14,14,1),name="x_view1")
encoder_inputs = keras.Input(shape=(1),name="x_view1")
#x = layers.Conv2D(32,3,activation='relu',strides=2,padding="same")(encoder_inputs)
#x = layers.Conv2D(64,3,activation='relu',strides=2,padding="same")(x)
#x = layers.Flatten()(x)
x = layers.Dense(4, activation="relu")(encoder_inputs)
x = layers.Dense(4, activation="relu")(x)
z_mean = layers.Dense(latent_dim,name='z_mean')(x)
z_log_var = layers.Dense(latent_dim,name="z_log_var")(x)
z = Sampling()([z_mean,z_log_var])
encoder = keras.Model(encoder_inputs,[z_mean,z_log_var,z],name="encoder")
encoder.summary()

#encoder_inputs_v2 = keras.Input(shape=(14,14,1),name="x_view2")
encoder_inputs_v2 = keras.Input(shape=(1),name="x_view2")
#x = layers.Conv2D(32,3,activation='relu',strides=2,padding="same")(encoder_inputs_v2)
#x = layers.Conv2D(64,3,activation='relu',strides=2,padding="same")(x)
#x = layers.Flatten()(x)
x = layers.Dense(4,activation="relu")(encoder_inputs_v2)
x = layers.Dense(4,activation="relu")(x)
z_mean_v2 = layers.Dense(latent_dim_v2,name="z_mean_v2")(x)
z_log_var_v2 = layers.Dense(latent_dim_v2,name="z_log_var_v2")(x)
zv2 = Sampling()([z_mean_v2,z_log_var_v2])
encoder_v2 = keras.Model(encoder_inputs_v2,[z_mean_v2,z_log_var_v2,zv2],name="encoder_v2")
encoder_v2.summary()


consensus_input_v1 = keras.Input(shape=(latent_dim,))
consensus_input_v2 = keras.Input(shape=(latent_dim_v2,))
z_merge = layers.concatenate([consensus_input_v1,consensus_input_v2])
x = layers.Dense(4,activation="relu")(z_merge) # shared latent representation
z_shared_mean = layers.Dense(shared_latent_dim,name="z_mean_shared")(x)
z_shared_log_var = layers.Dense(shared_latent_dim,name="z_log_var_shared")(x)
z_shared = Sampling()([z_shared_mean,z_shared_log_var])
consensus = keras.Model([consensus_input_v1,consensus_input_v2],[z_shared_mean,z_shared_log_var,z_shared],name="consensus")
consensus.summary()


latent_inputs = keras.Input(shape=(shared_latent_dim,))
#x = layers.Dense(64,activation=None)(latent_inputs)
decoder_output = layers.Dense(nclass,activation=None)(latent_inputs) # linear decoder
decoder = keras.Model(latent_inputs,decoder_output, name="decoder")
decoder.summary()


class DMVIB(keras.Model):
	def __init__(self,encoder,encoder_v2,consensus,decoder,gamma_v1,gamma_v2,**kwargs):
		super(DMVIB,self).__init__(**kwargs)
		self.encoder = encoder
		self.encoder_v2 = encoder_v2
		self.consensus = consensus
		self.decoder = decoder
		self.gamma_v1 = gamma_v1
		self.gamma_v2 = gamma_v2
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.ce_loss_tracker = keras.metrics.Mean(name="crossentropy_loss")
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
		self.kl_loss_v2_tracker = keras.metrics.Mean(name="kl_loss")
		self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.test_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
		self.acc_tracker = keras.metrics.SparseCategoricalAccuracy()
	@property
	def metrics(self):
		return [self.total_loss_tracker,self.ce_loss_tracker,self.kl_loss_tracker,self.kl_loss_v2_tracker,self.acc_tracker]
	def train_step(self,data):
		(dict_x,dict_y) = data
		#print(t1.shape,t2.shape)# get dictionary
		x_data = dict_x["x_view1"]
		x_data_v2 = dict_x["x_view2"]
		y_data = dict_y["decoder"]
		#(x_data,x_data_v2,y_data) = data
		with tf.GradientTape() as tape:
			z_mean,z_log_var,z = self.encoder(x_data)
			z_mean_v2,z_log_var_v2,z_v2 = self.encoder_v2(x_data_v2)
			_,_,z_merge = self.consensus([z,z_v2])
			logits = self.decoder(z_merge)

			ce_loss_val = self.ce_loss(y_data,logits)
			kl_loss_v1 = -0.5*(1+z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
			kl_loss_v1 = tf.reduce_mean(tf.reduce_sum(kl_loss_v1, axis=1))
			kl_loss_v2 = -0.5*(1+z_log_var_v2 - tf.square(z_mean_v2) - tf.exp(z_log_var_v2))
			kl_loss_v2 = tf.reduce_mean(tf.reduce_sum(kl_loss_v2, axis=1))
			total_loss = self.gamma_v1* kl_loss_v1 + self.gamma_v2 * kl_loss_v2 + ce_loss_val
		grads = tape.gradient(total_loss,self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.ce_loss_tracker.update_state(ce_loss_val)
		self.kl_loss_tracker.update_state(kl_loss_v1)
		self.kl_loss_v2_tracker.update_state(kl_loss_v2)
		self.acc_tracker.update_state(y_data,logits)
		return {"loss":self.total_loss_tracker.result(),"ce_loss":self.ce_loss_tracker.result(),
				"kl_loss_v1":self.kl_loss_tracker.result(),"kl_loss_v2":self.kl_loss_v2_tracker.result(),
				"accuracy":self.acc_tracker.result()}
	def test_step(self,data):
		
		(dict_x,dict_y) = data
		x_data = dict_x["x_view1"]
		x_data_v2 = dict_x["x_view2"]
		y_data = dict_y["decoder"]

		z_mean,z_log_var,z = self.encoder(x_data)
		z_mean_v2,z_log_var_v2,z_v2 = self.encoder_v2(x_data_v2)
		_,_,z_merge = self.consensus([z,z_v2])
		logits = self.decoder(z_merge)
		ce_loss_val = self.ce_loss(y_data,logits)
		kl_loss_v1 = -0.5*(1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var))
		kl_loss_v1 = tf.reduce_mean(tf.reduce_sum(kl_loss_v1, axis=1))
		kl_loss_v2 = -0.5*(1+z_log_var_v2 - tf.square(z_mean_v2)-tf.exp(z_log_var_v2))
		kl_loss_v2 = tf.reduce_mean(tf.reduce_sum(kl_loss_v2, axis=1))
		total_loss = self.gamma_v1 * kl_loss_v1 + self.gamma_v2*kl_loss_v2 + ce_loss_val
		self.test_acc_tracker.update_state(y_data,logits)
		#return {"loss":self.total_loss_tracker.result(),"ce_loss":self.ce_loss_tracker.result(),
		#		"kl_loss_v1":self.kl_loss_tracker.result(),"kl_loss_v2":self.kl_loss_v2_tracker.result(),
		#		"accuracy":self.test_acc_tracker.result()}
		return {"accuracy":self.test_acc_tracker.result()}

'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train,-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test,-1).astype("float32") / 255.0
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# separate into two views... how...
x_train_v1 = x_train[:,0:14,0:14]
x_train_v2 = x_train[:,14:,14:]
x_test_v1 = x_test[:,0:14,0:14]
x_test_v2 = x_test[:,14:,14:]
#x_test_v1 = x_test[:,0:14,0:14]
#x_test_v2 = x_test[:,14:,14:]
'''
with open('traindata_syn2_simple_num_60000.pkl','rb') as fid:
	mytraindata = pickle.load(fid)
x_train = mytraindata['x_test'].astype("float32")
y_train = mytraindata['y_test'].astype("float32")
x_min = np.amax(x_train,axis=0)
x_train /= x_min

x_train_v1 = x_train[:,0]
x_train_v2 = x_train[:,1]

with open('testdata_syn2_simple_num_10000.pkl','rb') as fid:
	mytestdata = pickle.load(fid)
x_test = mytestdata['x_test'].astype("float32")
y_test = mytestdata['y_test'].astype("float32")
x_test /= x_min # from training

x_test_v1 = x_test[:,0]
x_test_v2 = x_test[:,1]



# Prepare the validation dataset
#val_dataset = tf.data.Dataset.from_tensor_slices((x_test_v1,x_test_v2, y_test))
#val_dataset = val_dataset.batch(100)
batchsize = 1024
val_dataset = tf.data.Dataset.from_tensor_slices((
		{"x_view1":x_test_v1, "x_view2":x_test_v2},
		{"decoder":y_test},)
	)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"x_view1": x_train_v1, "x_view2": x_train_v2},
        {"decoder": y_train},
    )
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batchsize)
val_data = val_dataset.batch(batchsize)


#x_trainmerge = zip(x_train_v1,x_train_v2,y_train)
#x_testmerge = zip(x_test_v1,x_test_v2,y_test)
mvib = DMVIB(encoder,encoder_v2,consensus,decoder,gamma_v1,gamma_v2)
mvib.compile(optimizer=keras.optimizers.Adam())
#mvib.fit({"x_view1":x_train_v1,"x_view2":x_train_v2},{"decoder":y_train},
#		validation_data=val_dataset,epochs=30,batch_size=128)
#mvib.fit({"x_view1":x_train_v1,"x_view2":x_train_v2},{"decoder":y_train},
#		validation_data=(x_test,y_test),epochs=10,batch_size=128)
mvib.fit(train_dataset,epochs=30,batch_size=batchsize)
# FIXME: seems like we can only manually test the model
test_ce_loss_tracker = tf.keras.metrics.Mean(name="test_ce_loss")
test_kl_lossv1_tracker = tf.keras.metrics.Mean(name="test_kl_loss_v1")
test_kl_lossv2_tracker = tf.keras.metrics.Mean(name="test_kl_loss_v2")
test_loss_tracker = tf.keras.metrics.Mean(name="test_total_loss")

for xdict,ydict in val_data:
	x_data = xdict['x_view1']
	x_data_v2 = xdict['x_view2']
	y_data = ydict['decoder']
	#print(x_data.shape)
	#print(x_data_v2.shape)
	z_mean,z_log_var,z = mvib.encoder(x_data)
	z_mean_v2,z_log_var_v2,z_v2 = mvib.encoder_v2(x_data_v2)
	_,_,z_merge = mvib.consensus([z,z_v2])
	logits = mvib.decoder(z_merge)
	ce_loss_val = mvib.ce_loss(y_data,logits)
	kl_loss_v1 = -0.5*(1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var))
	kl_loss_v1 = tf.reduce_mean(tf.reduce_sum(kl_loss_v1, axis=1))
	kl_loss_v2 = -0.5*(1+z_log_var_v2 - tf.square(z_mean_v2)-tf.exp(z_log_var_v2))
	kl_loss_v2 = tf.reduce_mean(tf.reduce_sum(kl_loss_v2, axis=1))
	total_loss = gamma_v1 * kl_loss_v1 + gamma_v2*kl_loss_v2 + ce_loss_val
	mvib.test_acc_tracker.update_state(y_data,logits)
	test_ce_loss_tracker.update_state(ce_loss_val)
	test_kl_lossv1_tracker.update_state(kl_loss_v1)
	test_kl_lossv2_tracker.update_state(kl_loss_v2)
	test_loss_tracker.update_state(total_loss)
print('TESTING: gamma={:>8.4f}, Accuracy={:>8.4f}, Total loss={:>8.4f}, KL_v1={:>8.4f}, KL_v2={:>8.4f}, CE={:8.4f}'.format(
		gamma_v1,		
		mvib.test_acc_tracker.result(), test_loss_tracker.result(),
		test_kl_lossv1_tracker.result(),test_kl_lossv2_tracker.result(),
		test_ce_loss_tracker.result(),
	))
