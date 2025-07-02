import numpy as np
import pandas as pd
from sklearn import preprocessing
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


with tf.device('/gpu:1'):

	################################## Load Robot data ##################################################################
	Arm2_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_new.csv', header=None)
	Arm2_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_new.csv', header=None)

	robot_state_train_input = Arm2_CS_State[0:50244][:]
	print("Robot state input trainingset size: {}".format(robot_state_train_input.shape))
	robot_state_train_label = Arm2_NS_State[0:50244][:]
	print("Robot state label trainingset size: {}".format(robot_state_train_label.shape))

	robot_state_test_input = Arm2_CS_State[50244:][:]
	print("Robot state input testset size: {}".format(robot_state_test_input.shape))
	robot_state_test_label = Arm2_NS_State[50244:][:]
	print("Robot state label testset size: {}".format(robot_state_test_label.shape))

	################################## Standardization ###################################################################
	CS_train_names = robot_state_train_input.columns
	NS_train_names = robot_state_train_label.columns

	CS_test_names = robot_state_test_input.columns
	NS_test_names = robot_state_test_label.columns

	scaler = preprocessing.StandardScaler()
	input_Scaler = scaler.fit(robot_state_train_input)
	output_Scaler = scaler.fit(robot_state_train_label)
	robot_state_train_input = input_Scaler.transform(robot_state_train_input)
	robot_state_train_label = output_Scaler.transform(robot_state_train_label)

	robot_state_test_input = input_Scaler.transform(robot_state_test_input)
	robot_state_test_label = output_Scaler.transform(robot_state_test_label)

	robot_state_train_input = pd.DataFrame(robot_state_train_input, columns=CS_train_names)
	robot_state_train_label = pd.DataFrame(robot_state_train_label, columns=NS_train_names)

	robot_state_test_input = pd.DataFrame(robot_state_test_input, columns=CS_test_names)
	robot_state_test_label = pd.DataFrame(robot_state_test_label, columns=NS_test_names)

	robot_state_train_input = np.array(robot_state_train_input)
	robot_state_train_label = np.array(robot_state_train_label)

	robot_state_test_input = np.array(robot_state_test_input)
	robot_state_test_label = np.array(robot_state_test_label)

	############################################### Load image data #####################################################
	X_train_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/trainImageName.csv', header=None)
	X_test_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/testImageName.csv', header=None)
	X_train_filenames = np.array(X_train_filenames)
	X_test_filenames = np.array(X_test_filenames)

	X_train_filenames = X_train_filenames[:, 0]
	X_test_filenames = X_test_filenames[:, 0]

	######################################################################################################################
	class My_Custom_Generator(keras.utils.Sequence):
		def __init__(self, image_filenames, robot_input, labels, batch_size):
			self.image_filenames = image_filenames
			self.robot_input = robot_input
			self.labels = labels
			self.batch_size = batch_size
		
		def __len__(self):
			return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
		
		def __getitem__(self, idx):
			batch_x_img = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
			batch_x_robot = self.robot_input[idx * self.batch_size : (idx+1) * self.batch_size]
			batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
			
			# Process images
			images = np.array([
				resize(imread('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/all_images/' + str(file_name)), (224, 224, 3))
				for file_name in batch_x_img
			], dtype=np.float32) / 255.0
			
			
			robot_data = np.array(batch_x_robot, dtype=np.float32)
			labels = np.array(batch_y, dtype=np.float32)
			
			# Return as list since model expects inputs=[image_input_layer, robot_state_input_layer]
			return {'image_input': images, 'robot_input': robot_data}, labels

	batch_size = 32

	my_training_batch_generator = My_Custom_Generator(X_train_filenames, robot_state_train_input, robot_state_train_label, batch_size)
	my_testing_batch_generator = My_Custom_Generator(X_test_filenames, robot_state_test_input, robot_state_test_label, batch_size)
	
	#######################################################################################################################################
	###################################### Define CNN ##############################

	image_input_layer = keras.layers.Input(shape=(224,224,3), name='image_input')

	layer_conv_1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(image_input_layer)
	layer_conv_2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_1)
	layer_pooling_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_2)

	layer_conv_3 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_1)
	layer_conv_4 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_3)
	layer_pooling_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_4)

	layer_conv_5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_2)
	layer_conv_6 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_5)
	layer_conv_7 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_6)
	layer_pooling_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_7)

	layer_conv_8 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_3)
	layer_conv_9 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_8)
	layer_conv_10 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_9)
	layer_pooling_4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_10)

	layer_conv_11 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_4)
	layer_conv_12 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_11)
	layer_conv_13 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_12)

	cnn_flatten = keras.layers.Flatten()(layer_conv_13)

	robot_state_input_layer = keras.layers.Input(shape=(7,), name='robot_input')

	dense_1 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
	dense_2 = keras.layers.Dense(25, activation="relu")(dense_1)

	concat = keras.layers.concatenate([dense_2, cnn_flatten])

	dense_3 = keras.layers.Dense(80, activation="relu")(concat)
	dense_4 = keras.layers.Dense(20, activation="relu")(dense_3)
	output_layer = keras.layers.Dense(7, activation="linear")(dense_4)

	model = keras.models.Model(inputs=[image_input_layer, robot_state_input_layer], outputs=output_layer)

	# Define intermediate layer model for latent space extraction
	intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_3)

	##########################################################################################################################################

	# Early stopping callback
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, 
	        verbose=1, mode='auto', restore_best_weights=True)

	# Define optimizer and loss function
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	loss_fn = tf.keras.losses.MeanAbsoluteError()  # Changed to MAE for consistency with regression

	# Compile model
	model.compile(optimizer=opt, loss=loss_fn, metrics=['mse'])

	# Train the model
	print("Starting training...")
	print(f"Steps per epoch: {int(50244 // batch_size)}")
	
	history = model.fit(
		my_training_batch_generator, 
		steps_per_epoch=int(50244 // batch_size), 
		epochs=7,
		validation_data=my_testing_batch_generator,
		validation_steps=int(len(X_test_filenames) // batch_size),
		callbacks=[monitor],
		verbose=1
	)

	# Save models
	print("Saving models...")
	model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Customized_CNN.h5')
	intermediate_layer_model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/CustomizedCNN_intermediate_layer.h5')
	
	print("Training completed and models saved successfully!")