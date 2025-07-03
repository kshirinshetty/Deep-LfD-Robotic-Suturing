# pyright: reportMissingImports=false
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

from skimage.io import imread
from skimage.transform import resize

# Allow GPU memory growth (optional but improves GPU handling)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth setting failed:", e)

np.random.seed(1000)

with tf.device('/gpu:0'):

    # Load Robot Data
    Arm2_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_fps3.csv', header=None)
    Arm2_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_fps3.csv', header=None)

    robot_state_train_input = Arm2_CS_State[0:10000]
    robot_state_train_label = Arm2_NS_State[0:10000]
    robot_state_test_input = Arm2_CS_State[10000:]
    robot_state_test_label = Arm2_NS_State[10000:]

    print("Train input:", robot_state_train_input.shape)
    print("Train label:", robot_state_train_label.shape)
    print("Test input:", robot_state_test_input.shape)
    print("Test label:", robot_state_test_label.shape)

    # Standardization
    input_scaler = preprocessing.StandardScaler().fit(robot_state_train_input)
    output_scaler = preprocessing.StandardScaler().fit(robot_state_train_label)

    robot_state_train_input = input_scaler.transform(robot_state_train_input)
    robot_state_train_label = output_scaler.transform(robot_state_train_label)
    robot_state_test_input = input_scaler.transform(robot_state_test_input)
    robot_state_test_label = output_scaler.transform(robot_state_test_label)

    # Convert to NumPy arrays with explicit dtype
    robot_state_train_input = np.array(robot_state_train_input, dtype=np.float32)
    robot_state_train_label = np.array(robot_state_train_label, dtype=np.float32)
    robot_state_test_input = np.array(robot_state_test_input, dtype=np.float32)
    robot_state_test_label = np.array(robot_state_test_label, dtype=np.float32)

    # Load image filenames
    X_train_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/filtered/train_filtered.csv', header=None).values[:, 0]
    X_test_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/filtered/test_filtered.csv', header=None).values[:, 0]

    # Custom data generator
    class MyCustomGenerator(keras.utils.Sequence):
        def __init__(self, image_filenames, robot_input, labels, batch_size):
            self.image_filenames = image_filenames
            self.robot_input = robot_input
            self.labels = labels
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.image_filenames) / self.batch_size))

        def __getitem__(self, idx):
            batch_x_img = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_x_robot = self.robot_input[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

            images = []
            for file_name in batch_x_img:
                img_path = f'/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/all_images/{file_name}'
                img = imread(img_path)
                img = resize(img, (224, 224, 3), anti_aliasing=True)
                images.append(img)

            images = np.array(images, dtype=np.float32) / 255.0
            robot_data = np.array(batch_x_robot, dtype=np.float32)
            labels = np.array(batch_y, dtype=np.float32)

            return (images, robot_data), labels

    batch_size = 64
    train_gen = MyCustomGenerator(X_train_filenames, robot_state_train_input, robot_state_train_label, batch_size)
    test_gen = MyCustomGenerator(X_test_filenames, robot_state_test_input, robot_state_test_label, batch_size)

    # Define the model
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')(image_input)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, (11, 11), activation='relu', padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)

    robot_input = Input(shape=(7,), name='robot_input')
    r = Dense(15, activation='relu')(robot_input)
    r = Dense(25, activation='relu')(r)

    merged = concatenate([x, r])
    z = Dense(80, activation='relu')(merged)
    z = Dense(20, activation='relu')(z)
    output = Dense(7, activation='linear')(z)

    model = keras.Model(inputs=[image_input, robot_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])

    # Print model summary and parameter count
    model.summary()
    trainable_count = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"\nTrainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")

    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, restore_best_weights=True)

    # Train the model
    start = time.time()
    history = model.fit(
        train_gen,
        epochs=7,
        callbacks=[early_stop],
        validation_data=test_gen,
        verbose=1
    )
    end = time.time()
    print(f"\n⏱️ Training took {(end - start):.2f} seconds.")

    # Save the model
    model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/AlexNet.h5')
    print("✅ Model training completed and saved successfully!")
