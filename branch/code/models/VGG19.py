import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn import preprocessing
from skimage.io import imread
from skimage.transform import resize

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
			return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

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

			# Return as dict for multi-input model with correct input names
			return {'input_layer': images, 'input_layer_1': robot_data}, labels

	batch_size = 32

	my_training_batch_generator = My_Custom_Generator(X_train_filenames, robot_state_train_input, robot_state_train_label, batch_size)
	my_testing_batch_generator = My_Custom_Generator(X_test_filenames, robot_state_test_input, robot_state_test_label, batch_size)
	#######################################################################################################################################
	##################################### Define CNN architecture #######################################################################
	# Explicitly name the input layers to match the generator keys
	image_input_layer = keras.layers.Input(shape=(224, 224, 3), name='input_layer')
	robot_state_input_layer = keras.layers.Input(shape=(7,), name='input_layer_1')

	base_vgg = VGG19(include_top=False, weights='imagenet', input_tensor=image_input_layer)
	y1 = base_vgg.output
	y2 = GlobalAveragePooling2D()(y1)
	y3 = Dense(512, activation='relu')(y2)
	y4 = Dense(512, activation='relu')(y3)

	new_model = Model(inputs=base_vgg.input, outputs=y4)

	for layer in base_vgg.layers[:21]:
		layer.trainable = False
	for layer in base_vgg.layers[21:]:
		layer.trainable = True

	cnn_out = new_model.output

	dense_1 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
	dense_2 = keras.layers.Dense(25, activation="relu")(dense_1)

	concat = keras.layers.concatenate([dense_2, cnn_out])

	dense_3 = keras.layers.Dense(80, activation="relu")(concat)
	dense_4 = keras.layers.Dense(20, activation="relu")(dense_3)
	output_layer = keras.layers.Dense(7, activation="linear")(dense_4)

	# Use the same input names as in the generator dict: 'input_layer' and 'input_layer_1'
	vgg19_model = keras.models.Model(inputs=[image_input_layer, robot_state_input_layer], outputs=output_layer)

	#####################################################################################################################
	############################## compile and fit the model #########################################################

	vgg19_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse', 'accuracy'])

	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2,
	        verbose=1, mode='auto', restore_best_weights=True)

	history = vgg19_model.fit(
	    my_training_batch_generator,
	    steps_per_epoch=len(my_training_batch_generator),
	    validation_data=my_testing_batch_generator,
	    validation_steps=len(my_testing_batch_generator),
	    callbacks=[monitor],
	    epochs=1
	)

	vgg19_model.summary()

	vgg19_model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/all_images/VGG19.h5')