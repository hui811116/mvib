import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
		return z_mean + tf.exp(0.5*z_log_var) * epsilon

latent_dim = 2
nclass = 10
gamma = 0.1

encoder_inputs = keras.Input(shape=(28,28,1))
x = layers.Conv2D(32,3,activation='relu',strides=2,padding="same")(encoder_inputs)
x = layers.Conv2D(64,3,activation='relu',strides=2,padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim,name='z_mean')(x)
z_log_var = layers.Dense(latent_dim,name="z_log_var")(x)
z = Sampling()([z_mean,z_log_var])
encoder = keras.Model(encoder_inputs,[z_mean,z_log_var,z],name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
#x = layers.Dense(64,activation=None)(latent_inputs)
decoder_output = layers.Dense(nclass,activation=None)(latent_inputs) # linear decoder
decoder = keras.Model(latent_inputs,decoder_output, name="decoder")
decoder.summary()

class VIB(keras.Model):
	def __init__(self,encoder,decoder,gamma,**kwargs):
		super(VIB,self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.gamma = gamma
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.ce_loss_tracker = keras.metrics.Mean(name="crossentropy_loss")
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
		self.acc_tracker = keras.metrics.SparseCategoricalAccuracy()
		self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.test_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
	@property
	def metrics(self):
		return [self.total_loss_tracker,self.ce_loss_tracker,self.kl_loss_tracker,self.acc_tracker]
	def train_step(self, data):
		(x_data,y_data) = data
		with tf.GradientTape() as tape:
			z_mean,z_log_var, z= self.encoder(x_data)
			logits = self.decoder(z)
			ce_loss_val = self.ce_loss(y_data,logits)
			kl_loss = -0.5* (1+z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))
			total_loss = self.gamma * kl_loss + ce_loss_val

		grads= tape.gradient(total_loss,self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.ce_loss_tracker.update_state(ce_loss_val)
		self.kl_loss_tracker.update_state(kl_loss)
		self.acc_tracker.update_state(y_data,logits)
		return {"loss":self.total_loss_tracker.result(), "ce_loss":self.ce_loss_tracker.result(), "kl_loss":self.kl_loss_tracker.result(),"accuracy":self.acc_tracker.result()}
	def test_step(self,data):
		(x_test,y_test) = data
		z_mean,z_log_var,z = self.encoder(x_test)
		logits = self.decoder(z)
		ce_loss_val= self.ce_loss(y_test,logits)
		kl_loss = -0.5 * (1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var))
		total_loss = self.gamma * kl_loss + ce_loss_val
		self.test_acc_tracker.update_state(y_test,logits)
		return {"accuracy":self.test_acc_tracker.result()}


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train,-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test,-1).astype("float32") / 255.0
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255


vib = VIB(encoder, decoder,gamma)
vib.compile(optimizer=keras.optimizers.Adam())
vib.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=30, batch_size=128)
