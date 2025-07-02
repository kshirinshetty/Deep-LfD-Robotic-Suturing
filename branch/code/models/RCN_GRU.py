import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Configure GPU properly for laptop
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# GPU Configuration for laptop
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use the first GPU (index 0) for laptop
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Enable memory growth to prevent GPU memory issues
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
        print("Falling back to CPU")
else:
    print("No GPU found, using CPU")

# Remove the specific GPU device context - let TensorFlow handle it automatically
# with tf.device('/gpu:1'):  # This was causing issues

try:
    print("Loading data...")
    Arm1_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_new.csv')
    Arm1_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_new.csv')

    #X = np.load('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20imageDataGray.npy')
    X = np.load('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Test1-20image.npy')

    print("Data loaded successfully")
    print("First row of Arm1_CS_State:")
    print(Arm1_CS_State[0:1])

    D3data = np.ones((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    D3data[ : , : , : , :] = X
    print(f"D3data shape: {D3data.shape}")

    # Prepare training data
    train = D3data[0:D3data.shape[0]-2000 , : , : , :]
    train = train.astype(np.float32)  # Use float32 instead of float64 for GPU efficiency
    print("Training dataset size: {}".format(train.shape[0]))

    # Normalize training data with better handling
    print("Normalizing training data...")
    for i in range(train.shape[0]):
        max_val = np.max(train[i, : , : , :])
        if max_val > 0:  # Avoid division by zero
            train[i, : , : , :] = train[i, : , : , :] / max_val
        if i % 1000 == 0:
            print(f"Normalized {i}/{train.shape[0]} training samples")

    robot_state_train_input = Arm1_CS_State[0:train.shape[0]].values.astype(np.float32)
    print("Robot state input training set size: {}".format(robot_state_train_input.shape))
    robot_state_train_label = Arm1_NS_State[0:train.shape[0]].values.astype(np.float32)
    print("Robot state label training set size: {}".format(robot_state_train_label.shape))

    # Prepare test data
    test = D3data[D3data.shape[0]-2000: , : , : , :]
    test = test.astype(np.float32)  # Use float32 for GPU efficiency

    print("Normalizing test data...")
    for i in range(test.shape[0]):
        max_val = np.max(test[i, : , : , :])
        if max_val > 0:  # Avoid division by zero
            test[i, : , : , :] = test[i, : , : , :] / max_val
        if i % 500 == 0:
            print(f"Normalized {i}/{test.shape[0]} test samples")

    robot_state_test_input = Arm1_CS_State[train.shape[0]:train.shape[0]+test.shape[0]].values.astype(np.float32)
    print("Robot state input test set size: {}".format(robot_state_test_input.shape))
    robot_state_test_label = Arm1_NS_State[train.shape[0]:train.shape[0]+test.shape[0]].values.astype(np.float32)
    print("Robot state label test set size: {}".format(robot_state_test_label.shape))

    # Load intermediate model
    print("Loading intermediate layer model...")
    try:
        intermediate_layer_model = load_model('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/CNN_intermediate_layer.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print("Generating intermediate outputs...")
    intermediate_output_train = intermediate_layer_model.predict([train, robot_state_train_input], batch_size=32, verbose=1)
    intermediate_output_test = intermediate_layer_model.predict([test, robot_state_test_input], batch_size=32, verbose=1)

    print(f"Intermediate output train shape: {intermediate_output_train.shape}")

    validation_set_size = 1500

    train_set = intermediate_output_train[0:intermediate_output_train.shape[0]-validation_set_size , : ]
    validation_set = intermediate_output_train[intermediate_output_train.shape[0]-validation_set_size: , : ]
    test_set = intermediate_output_test[: , :]

    timeWindow = 4

    print("Creating time series generators...")
    train_gen = keras.preprocessing.sequence.TimeseriesGenerator(train_set, train_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
    validation_gen = keras.preprocessing.sequence.TimeseriesGenerator(validation_set, validation_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
    test_gen = keras.preprocessing.sequence.TimeseriesGenerator(test_set, test_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)

    print(f"Train generator sample shape: {train_gen[0][0].shape}")
    print("Train set shape: {}".format(train_set.shape))

    # Create training matrices
    print("Creating training matrices...")
    train_matrix = np.zeros((train_set.shape[0]-timeWindow, timeWindow, 80), dtype=np.float32)
    for i in range(timeWindow, train_set.shape[0]):
        train_matrix[i-timeWindow, : , : ] = train_gen[i-timeWindow][0][0]

    validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, timeWindow, 80), dtype=np.float32)
    for i in range(timeWindow, validation_set.shape[0]):
        validation_matrix[i-timeWindow, : , : ] = validation_gen[i-timeWindow][0][0]

    test_matrix = np.zeros((test_set.shape[0]-timeWindow, timeWindow, 80), dtype=np.float32)
    for i in range(timeWindow, test_set.shape[0]):
        test_matrix[i-timeWindow, : , : ] = test_gen[i-timeWindow][0][0]

    # Create output matrices
    print("Creating output matrices...")
    output_train_matrix = np.zeros((train_set.shape[0]-timeWindow, 7), dtype=np.float32)
    for i in range(timeWindow, train_set.shape[0]):
        output_train_matrix[i-timeWindow , :] = robot_state_train_label[i-timeWindow:(i-timeWindow+1)]

    output_validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, 7), dtype=np.float32)
    for i in range(timeWindow, validation_set.shape[0]):
        idx = intermediate_output_train.shape[0] - validation_set_size + i - timeWindow
        output_validation_matrix[i-timeWindow, : ] = robot_state_train_label[idx:(idx+1)]

    output_test_matrix = np.zeros((test_set.shape[0]-timeWindow, 7), dtype=np.float32)
    for i in range(timeWindow, test_set.shape[0]):
        output_test_matrix[i - timeWindow, :] = robot_state_test_label[(i - timeWindow):(i - timeWindow + 1)]

    print("Output test matrix shape: {}".format(output_test_matrix.shape))

    # Early stopping callback
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
            verbose=1, mode='auto', restore_best_weights=True)

    # Build GRU model
    print("Building GRU model...")
    GRU_Model = keras.models.Sequential([
        keras.layers.GRU(150, return_sequences=True, input_shape=(timeWindow, 80)),
        keras.layers.Dropout(0.2),  # Add dropout for regularization
        keras.layers.GRU(100),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(7, activation='linear')
    ])
    
    # Compile model with mixed precision for better GPU performance
    GRU_Model.compile(
        loss='mae', 
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        metrics=['mse']
    )
    
    print("Model architecture:")
    GRU_Model.summary()
    
    print("Starting model training...")
    
    # Train model with reduced batch size for stability
    history = GRU_Model.fit(
        train_matrix, output_train_matrix, 
        callbacks=[monitor], 
        epochs=40, 
        batch_size=32,  # Add batch size
        validation_data=(validation_matrix, output_validation_matrix),
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    score = GRU_Model.evaluate(test_matrix, output_test_matrix, verbose=0)
    print(f"Test loss: {score[0]:.4f}, Test MSE: {score[1]:.4f}")

    # Make predictions
    print("Making predictions...")
    test_predict_gru = GRU_Model.predict(test_matrix, batch_size=32)

    # Calculate errors
    err_GRU_Model = test_predict_gru - output_test_matrix
    Gru_err_mean = np.mean(np.abs(err_GRU_Model))
    print(f"Mean absolute error: {Gru_err_mean:.6f}")
    
    high_error_indices = np.where(np.abs(err_GRU_Model) > 0.01)
    high_error_count = len(high_error_indices[0])
    print(f"Number of error elements higher than 0.01: {high_error_count}")

    # Save model
    print("Saving model...")
    GRU_Model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/RCN_GRU.h5')
    print("Model saved successfully!")
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Script execution completed.")
    # Clear GPU memory
    if 'GRU_Model' in locals():
        del GRU_Model
    if 'intermediate_layer_model' in locals():
        del intermediate_layer_model
    tf.keras.backend.clear_session()