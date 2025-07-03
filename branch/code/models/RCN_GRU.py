# pyright: reportMissingImports=false
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os
import time

# Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU")

try:
    print("Loading data...")
    Arm1_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_new.csv')
    Arm1_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_new.csv')
    X = np.load('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Test1-20image.npy')

    print("Data loaded. Image shape:", X.shape)
    D3data = X.astype(np.float32, copy=False)

    # Split
    train = D3data[:-2000]
    test = D3data[-2000:]

    print("Normalizing train/test...")
    train_max = np.max(train, axis=(1, 2, 3), keepdims=True)
    train_max[train_max == 0] = 1
    train /= train_max

    test_max = np.max(test, axis=(1, 2, 3), keepdims=True)
    test_max[test_max == 0] = 1
    test /= test_max

    # Robot inputs/labels
    robot_state_train_input = Arm1_CS_State.iloc[:train.shape[0]].values.astype(np.float32)
    robot_state_train_label = Arm1_NS_State.iloc[:train.shape[0]].values.astype(np.float32)
    robot_state_test_input = Arm1_CS_State.iloc[train.shape[0]:train.shape[0]+test.shape[0]].values.astype(np.float32)
    robot_state_test_label = Arm1_NS_State.iloc[train.shape[0]:train.shape[0]+test.shape[0]].values.astype(np.float32)

    # Load model
    print("Loading intermediate model...")
    intermediate_layer_model = load_model('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/CNN_intermediate_layer.h5')

    print("Generating intermediate outputs...")
    intermediate_output_train = intermediate_layer_model.predict([train, robot_state_train_input], batch_size=32, verbose=1)
    intermediate_output_test = intermediate_layer_model.predict([test, robot_state_test_input], batch_size=32, verbose=1)

    # Time series windowing
    validation_set_size = 1500
    timeWindow = 4

    train_set = intermediate_output_train[:-validation_set_size]
    validation_set = intermediate_output_train[-validation_set_size:]
    test_set = intermediate_output_test

    def create_timeseries_matrix(data, output=None):
        samples = data.shape[0] - timeWindow
        X_out = np.zeros((samples, timeWindow, data.shape[1]), dtype=np.float32)
        for i in range(samples):
            X_out[i] = data[i:i+timeWindow]
        if output is not None:
            Y_out = output[timeWindow:]
            return X_out, Y_out
        return X_out

    train_matrix, output_train_matrix = create_timeseries_matrix(train_set, robot_state_train_label)
    validation_matrix, output_validation_matrix = create_timeseries_matrix(validation_set, robot_state_train_label[-validation_set_size:])
    test_matrix, output_test_matrix = create_timeseries_matrix(test_set, robot_state_test_label)

    print("Building GRU model...")
    GRU_Model = keras.models.Sequential([
        keras.layers.GRU(150, return_sequences=True, input_shape=(timeWindow, 80)),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(100),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(7, activation='linear')
    ])

    GRU_Model.compile(loss='mae', optimizer=keras.optimizers.Adam(1e-3), metrics=['mse'])
    GRU_Model.summary()

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, restore_best_weights=True)

    print("Training model...")
    history = GRU_Model.fit(
        train_matrix, output_train_matrix,
        validation_data=(validation_matrix, output_validation_matrix),
        callbacks=[monitor],
        epochs=40,
        batch_size=32,
        verbose=1
    )

    print("Evaluating...")
    loss, mse = GRU_Model.evaluate(test_matrix, output_test_matrix, verbose=0)
    print(f"Test loss: {loss:.4f}, Test MSE: {mse:.4f}")

    print("Predicting...")
    test_predict_gru = GRU_Model.predict(test_matrix, batch_size=32)
    err = test_predict_gru - output_test_matrix
    mean_abs_error = np.mean(np.abs(err))
    print(f"Mean absolute error: {mean_abs_error:.6f}")
    print("High error count (> 0.01):", np.sum(np.abs(err) > 0.01))

    print("Saving model...")
    GRU_Model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/RCN_GRU.h5')
    print("✅ Model saved!")

except Exception as e:
    print("❌ Error occurred:", e)
    import traceback
    traceback.print_exc()

finally:
    print("Script execution finished.")
    tf.keras.backend.clear_session()
