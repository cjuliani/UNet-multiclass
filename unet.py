import numpy as np
import tensorflow as tf
import os,layers,datasets,json,ast,util
from PIL import Image
from numpy.random import seed
import matplotlib as mpl
import matplotlib.pyplot as plt
seed(10)
tf.compat.v1.disable_eager_execution()

class unet(object):
	"""
	A unet implementation
	:param channels: number of channels in the input image
	:param n_class: number of output labels
	:param cost: (optional) name of the cost function. Default is 'cross_entropy'
	:param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
	"""
	def __init__(self):
		# reset default graph
		tf.compat.v1.reset_default_graph()
		
		# import pre-defined parameters
		with open('config.json') as file:
			params = json.load(file)
			
		# get parameters
		self.img_size = params['img_size']
		self.n_class = params['classes']									# number of classes
		self.in_dim = params['input_dimension']								# input dimension
		self.w_class = ast.literal_eval(params['class_weights'])			# class coef. for weighted loss | to be used if class imbalance
		self.tot_epoch = params['epoch']									# maximum number of training epochs
		self.batch_size = params['batch_size']								# batch size for training
		self.lr = params['learning_rate']									# learning rate
		self.dropout = params['dropout_rate']								# dropout rate
		self.progress = params['steps_to_display']							# averaged metrics to show after X steps | to be used if large dataset
		self.loss_thr = params['loss_threshold']							# loss value below which early stop is considered
		self.valid_loss_thr = params['valid_threshold']						# loss value below which early stop is considered (validation)
		self.loss_diff = params['loss_difference']							# loss difference above which early stop is considered
		self.early_stop = params['steps_to_stop']							# number of epochs to trigger early stop if loss_diff do not decrease
		self.chkpt_path = params['checkpoint']								# checkpoint folder path
		self.res_name = params['model_to_restore']							# model name to be restored | to be used for continuing or refining previous training
		self.model_path = params['model_path']								# path of model to restore
		self.restore = ast.literal_eval(params['restore_model'])			# boolean | restore model if True
		self.predict_path = params['prediction_path']						# path of prediction folder
		self.train_path = params['train_path']								# path of training folder

		with tf.name_scope('Placeholders'):
			self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, self.in_dim], name='input')
			self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.n_class], name='y_true')
			self.rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
			self.learning_rate_ = tf.compat.v1.placeholder(tf.float32, shape=())

		self.logits = layers.build_UNet(input=self.x,is_training=True,n_class=self.n_class,drop_rate=self.rate)

		with tf.name_scope('Cost'):
			# Calculate metrics
			self.cost, self.f1_vec, self.rec, self.prec, self.spec, self.acc = self.loss(logits=self.logits)

			# Create scalar summaries to be fetched in FileWriter
			loss_summary = tf.summary.scalar(name="loss", tensor=self.cost)
			for i in range(self.f1_vec.get_shape()[0]):
				f1_summary = tf.summary.scalar("f1_v", self.f1_vec[i])
			acc_summary = tf.summary.scalar("acc_m", self.acc)
			rec_summary = tf.summary.scalar("rec_m", self.rec)
			prec_summary = tf.summary.scalar("prec_m", self.prec)
			spec_summary = tf.summary.scalar("spec_m", self.spec)

		# Merge all summaries
		self.merged = tf.summary.merge_all()
		
		# Save variables to checkpoint files
		# Note: if no argument passed to .Saver(), saver handles all variables in the graph
		self.saver = tf.train.Saver()

	def loss(self, logits):
		''' Calculates cross-entropy loss
		:param class_weights: list of coef. | to be used for class imbalance'''
		assert (len(logits) == len(self.w_class)), "Number of classes ({0}) iand weight coefficients ({1}) not matching.".format(len(logits),len(self.w_class))

		epsilon = tf.constant(value=1e-10)
		with tf.name_scope('loss'):
			labels = tf.cast(tf.reshape(self.y_true, (-1, self.n_class)), tf.float32)
			cost, precision, recall, specificity, accuracy = 0.,0.,0.,0.,0.
			f1 = []
			for i in range(self.n_class):
				lgts = tf.reshape(logits[i], (-1, 2))
				lbls = tf.one_hot( tf.cast(labels[:,i], tf.int32), depth=2)

				cost += tf.math.divide( tf.reduce_mean( tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=lgts,labels=lbls), self.w_class[i] )), float(self.n_class))

				lgts_ = tf.argmax( tf.nn.softmax( lgts ), axis=1 )
				lbls_ = tf.argmax( lbls, axis=1 )

				TP = tf.compat.v1.count_nonzero( lgts_ * lbls_, dtype=tf.float32, axis=0)
				TN = tf.compat.v1.count_nonzero( (lgts_ - 1) * (lbls_ - 1), dtype=tf.float32, axis=0)
				FP = tf.compat.v1.count_nonzero( lgts_ * (lbls_ - 1), dtype=tf.float32, axis=0)
				FN = tf.compat.v1.count_nonzero( (lgts_ - 1) * lbls_, dtype=tf.float32, axis=0)
				# divide_no_NAN in case no TP exist in sample
				rec = tf.math.divide_no_nan( TP, (TP+FN) )
				prec = tf.math.divide_no_nan( TP, (TP+FP) )
				spec = tf.math.divide_no_nan( TN, (TN+FP) )
				acc = tf.math.divide_no_nan( (TP+TN), (TP+TN+FP+FN) )
				# divide by the number of classes to average the metrics
				accuracy += tf.math.divide( acc , float(self.n_class))
				recall += tf.math.divide( rec , float(self.n_class))
				precision += tf.math.divide( prec , float(self.n_class))
				specificity += tf.math.divide( spec , float(self.n_class))
				# store F1 scores
				f1_ = 2 * prec * rec / (prec + rec + epsilon)
				f1 += [f1_]

			f1 = tf.convert_to_tensor(f1, dtype=tf.float32)

		return cost, f1, recall, precision, specificity, accuracy

	def show_progress(self, epoch, loss, i, f1, rec, prec, spec, show):
		# display progression
		msg = "Epoch {0} --- i: {1} --- Loss: {2:.5f} --- f1: {3} --- rec: {4:.5f} --- prec: {5:.5f} --- acc: {6:.5f}"
		msg = msg.format(epoch, i, loss, f1, rec, prec, spec)
		if show == True:
			print(msg)

	def array_to_text(self, arr,n):
		''' 
		Convert F1 scores into string
		:param n: number of decimal digits
		'''
		txt = list(arr)
		if n == 3:
			txt = ["%.3f"%item for item in txt]
		else:
			txt = ["%.0f"%item for item in txt]
		return ' | '.join(e for e in txt)

	def save_parameters(self, filename,sess,saver,step):
		'''
		Save tensor parameters 
		'''
		self.saver.save(sess, os.path.join(self.chkpt_path, filename), global_step=step)

	def test(self,test_data,restore=False):

		n = test_data._num_examples
		print('\nNumber of inputs:', n)

		with tf.Session() as sess:

			# initialize variables
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			# restore model 
			if restore == True:
				# pre-trained
				model = os.path.join(self.model_path,self.res_name)
				self.saver.restore(sess, model)
				print('\nPre-trained model {} in folder {} restored.'.format(self.res_name,self.model_path))
			else:
				# currently training
				model = os.path.join(self.chkpt_path,self.res_name)
				self.saver.restore(sess, model)
				print('\nCurrent model {} in folder {} restored.'.format(self.res_name,self.chkpt_path))

			# initialize parameters
			avg_loss, avg_rec, avg_prec, avg_spec = 0.,0.,0.,0.
			avg_f1 = np.zeros((self.n_class,))
			for i in range(n):
				x_, y_true_, xn, _ = test_data.next_batch(1,data_aug=False)

				# get metrics
				feed_dict_tr = {self.x: x_, self.y_true: y_true_, self.rate: 0.}
				loss_ = sess.run(self.cost, feed_dict=feed_dict_tr)
				f1_ = sess.run(self.f1_vec, feed_dict=feed_dict_tr)
				f1_txt = self.array_to_text(f1_,3)
				rec_ = sess.run(self.rec, feed_dict=feed_dict_tr)
				prec_ = sess.run(self.prec, feed_dict=feed_dict_tr)
				spec_ = sess.run(self.spec, feed_dict=feed_dict_tr)
				#
				avg_loss += loss_/ n
				avg_rec += rec_/ n
				avg_prec += prec_/ n
				avg_spec += spec_/ n
				avg_f1 += f1_/ n
					
				# display results
				idx = xn[0][0].split('_')[-1].split('.')[0]
				print("Input {0:.0f}, Loss: {1:.3f} --- f1: {2} --- recall: {3:.3f} --- precision: {4:.3f} --- specificity: {5:.3f}".format(int(idx), loss_, f1_txt, rec_, prec_, spec_) )

		avg_f1_txt = self.array_to_text(avg_f1,3)
		print("\nAvg. loss: %.3f --- Avg. f1: " % (avg_loss) + avg_f1_txt)
		print("Avg. recall: {0:.3f} --- Avg. precision: {1:.3f} --- Avg. specificity: {2:.3f}".format(avg_rec, avg_prec, avg_spec) )

	def train(self,training_data,validation_data):
		with tf.Session() as sess:
			# Create writers for storing summaries
			train_writer = tf.summary.FileWriter(os.path.join(self.train_path, 'summary/training'), graph=sess.graph)
			validate_writer = tf.summary.FileWriter(os.path.join(self.train_path, 'summary/validation'))

			# Get optimizer
			optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.cost)

			# restore a pre-trained model
			if self.restore is True:
				# get model paths
				model = os.path.join(self.model_path,self.res_name)

				# initialize all variables
				sess.run(tf.global_variables_initializer())
				sess.run(tf.local_variables_initializer())

				# restore model
				self.saver.restore(sess, model)
			else:
				sess.run(tf.global_variables_initializer())
				sess.run(tf.local_variables_initializer())

			# Open file to write results
			# Note: this is to overwrite old files
			__ = open(self.train_path+'training.txt', 'w')
			__ = open(self.train_path+'validation.txt', 'w')

			# Initialize parameters for early-stop 
			stopping_step, best_loss, avg_loss_val = 0, 999., 999.

			# Get number of steps
			iterations = int(training_data._num_examples / self.batch_size)
			print('Number of epochs: ', self.tot_epoch )
			print('Number of steps per epoch: ', iterations)

			# Iterate training through epochs
			for epoch in range(1,self.tot_epoch+1):
				step_loss, step_rec, step_prec, step_spec, step_acc = 0.,0.,0.,0.,0.
				avg_loss, avg_rec, avg_prec, avg_spec, avg_acc = 0.,0.,0.,0.,0.
				avg_f1, step_f1 = np.zeros((self.n_class,)), np.zeros((self.n_class,))

				for i in range(iterations):
					# Get batches
					x_batch, y_true_batch, img_nm, _ = training_data.next_batch(self.batch_size,data_aug=True)
					feed_dict_tr = {self.x: x_batch, self.y_true: y_true_batch, self.rate: self.dropout, self.learning_rate_: self.lr}

					# Get metrics
					sess.run(optimizer, feed_dict=feed_dict_tr)
					loss_ = sess.run(self.cost, feed_dict=feed_dict_tr)
					f1_ = sess.run(self.f1_vec, feed_dict=feed_dict_tr)
					f1_txt = self.array_to_text(f1_,3)
					acc_ = sess.run(self.acc, feed_dict=feed_dict_tr)
					rec_ = sess.run(self.rec, feed_dict=feed_dict_tr)
					prec_ = sess.run(self.prec, feed_dict=feed_dict_tr)

					# Calculate and display metrics after every number of steps reached
					# Note: this is to keep track on progression if number of steps is very large
					step_loss += loss_/self.progress
					step_prec += prec_/self.progress
					step_rec += rec_/self.progress
					step_acc += acc_/self.progress
					step_f1 += f1_/self.progress
					if i % self.progress == 0 and i != 0:
						step_f1_txt = self.array_to_text(step_f1,3)
						self.show_progress(epoch, step_loss, i, step_f1_txt, step_rec, step_prec, step_acc, True)
						step_loss, step_rec, step_prec, step_acc = 0.,0.,0.,0.
						step_f1 = np.zeros((self.n_class,))

					# Get average metrics
					avg_loss += loss_/iterations
					avg_f1 += f1_/iterations
					avg_acc += acc_/iterations
					avg_rec += rec_/iterations
					avg_prec += prec_/iterations

					# Evaluate scalar summaries
					summary = sess.run(self.merged, feed_dict=feed_dict_tr)
					train_writer.add_summary(summary, i)
	                
					# Save model's parameters when an epoch starts and ends
					if i % int(training_data._num_examples/self.batch_size) == 0 or i == (iterations - 1):
						self.save_parameters(filename='segmentation.ckpt',sess=sess,saver=self.saver,step=i)

				'''
				Epoch tasks
				'''
				# Calculate and display average metrics 
				avg_f1_txt = self.array_to_text(avg_f1,3)
				msg = "\nEpoch {0} / {1} (lr_{2}) --- Avg_loss: {3:.5f} --- Avg_f1: {4} --- Avg_rec: {5:.5f} --- Avg_prec: {6:.5f} --- Avg_acc: {7:.5f}"
				print(msg.format(epoch, self.tot_epoch, self.lr, avg_loss, avg_f1_txt, avg_rec, avg_prec, avg_acc))
				# Write metrics
				cont = 'Epoch ' + str(epoch) + ": " + str(avg_loss) + "; f1: " + avg_f1_txt + "; rec_mean: " + str(avg_rec) + "; prec_mean: " + str(avg_prec) + "; acc_mean: " + str(avg_acc) + "\n"
				# Append new lines to opened text file
				# Note: append 'a' is more stable than write 'w' during the training phase
				with open(self.train_path+'training.txt', 'a') as file_a:
					file_a.write(cont)

				# Validation
				max_val_batch = validation_data._num_examples
				avg_f1_val, avg_rec_val, avg_prec_val, avg_spec_val, avg_acc_val, avg_loss_val = 0.,0.,0.,0.,0.,0.
				
				for j in range(max_val_batch):
					# Treat validation data individually
					x_valid, y_true_valid, _, _ = validation_data.next_batch(1,data_aug=False)

					# Get validation metrics
					feed_dict_vl = {self.x: x_valid, self.y_true: y_true_valid, self.rate: 0.}
					loss_val = sess.run(self.cost, feed_dict=feed_dict_vl)
					f1_val = sess.run(self.f1_vec, feed_dict=feed_dict_vl)
					rec_val = sess.run(self.rec, feed_dict=feed_dict_vl)
					prec_val = sess.run(self.prec, feed_dict=feed_dict_vl)
					acc_val = sess.run(self.acc, feed_dict=feed_dict_vl)

					# Calculate average metrics
					avg_loss_val += loss_val / max_val_batch
					avg_f1_val += f1_val / max_val_batch
					avg_rec_val += rec_val / max_val_batch
					avg_prec_val += prec_val / max_val_batch
					avg_acc_val += acc_val / max_val_batch

					# Evaluate scalar summaries
					summary = sess.run(self.merged, feed_dict=feed_dict_vl)
					validate_writer.add_summary(summary, j)
	            
	            # Display validation metrics
				f1_val_txt = self.array_to_text(avg_f1_val,3)
				msg = "Validation --- Avg. loss: {0:.5f} --- Avg_f1: {1} --- Avg_rec: {2:.5f} --- Avg_prec: {3:.5f} --- Avg_acc: {4:.5f}"
				print(msg.format(avg_loss_val, f1_val_txt, avg_rec_val, avg_prec_val, avg_acc_val))

				# Write metrics
				cont_val = 'Epoch ' + str(epoch) + ": " + str(avg_loss_val) + "; f1: " + f1_val_txt  + "; rec_mean: " + str(avg_rec_val) +"; prec_mean: " + str(avg_prec_val) + "; acc_mean: " + str(avg_acc_val) + "\n"
				with open(self.train_path+'validation.txt', 'a') as file_b:
					# Append new lines to opened text file
					# Note: append 'a' is more stable than write 'w' during the training phase
					file_b.write(cont_val)
	            
				'''
				Early stopping
				'''
				# Stop training over X epochs if the loss difference between trainning and validation X or above.
				# This stops only happens if the loss threshold for training / validation is less X and X respectively.

				# Loss difference
				loss_d = np.abs( avg_loss_val - avg_loss )

				# Conditions to stop
				if (stopping_step >= self.early_stop) and (avg_loss < self.loss_thr) and (avg_loss_val < self.valid_loss_thr) and (loss_d >= self.loss_diff):
					print("\n------")
					print("\nEarly stopping triggered at epoch: {0} --- loss: {1:.3f} --- validation: {2:.3f} --- loss difference: {3:.3f}".format(epoch,avg_loss,avg_loss_val,loss_d))
					self.save_parameters(filename='segmentation_earlystop.ckpt',sess=sess,saver=self.saver,step=epoch)
					# "return" to break multiple loops
					return
				# Keep track of loss difference
				if (loss_d < self.loss_diff):
					# if it keeps decreasing, reset epoch counter and continue
					stopping_step = 0
					best_loss = loss_d
				else:
					# else, increase epoch counter
					stopping_step += 1
				print('Loss difference: {0:.4f} --- stopping step: {1}\n'.format(loss_d,stopping_step))

	def predict(self,in_size,output=1):
		'''
		Predict instances
		:params in_size: size of input
		:params output: softmax output index
		'''

		# read inputs
		img_names = datasets.load_train(path=self.predict_path+"inputs")
		imgs = util.get_image_array(imgs=img_names,img_size=in_size)
		print('\nShape of input:', imgs.shape)

		# get logits for each class
		predictions = []
		for clss in range(self.n_class):
			lgt = tf.reshape(self.logits[clss], (-1, 2))		# reshape logits into vector representations
			lgt = tf.nn.softmax(lgt)							# calculate probabilities
			predictions.append(lgt)

		with tf.Session() as sess:

			# Initialize variables
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			# restore model
			model = os.path.join(self.model_path,self.res_name)
			self.saver.restore(sess, model)
			print('\nPre-trained model {} in folder {} restored.'.format(self.res_name,self.model_path))

			# get colormap
			cm_gradient = plt.get_cmap('jet')

			for i in range(imgs.shape[0]):

				# reshape each input
				imgs_ = np.expand_dims(imgs[i],axis=0)

				# get parent folder name
				folder_name = img_names[i][0].split('\\')[-2]
				
				# get feature maps
				feed_dict_tr = {self.x: imgs_,self.rate: 0.}
				for clss in range(self.n_class):
					result = sess.run(predictions[clss], feed_dict=feed_dict_tr)
					result = result.reshape(1,in_size,in_size,2)
					result = np.uint32(result[0][:,:,output] * 255)
					result = cm_gradient(result)
					result = Image.fromarray(np.uint8(result*255))
					
					# save predictions
					img_name = folder_name+'_input-'+str(i)+'_class-'+str(clss)+'_output-'+str(output)+'.png'
					output_name = os.path.join(self.predict_path, img_name)
					result.save(output_name)
					print('Saving '+img_name+'...')