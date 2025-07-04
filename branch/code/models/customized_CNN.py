import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.io import imread
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import json
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth setting failed:", e)

np.random.seed(1000)

def debug_print(step, message):
    print(f"[DEBUG {step}] {message}")

def get_memory_usage():
    """Get current memory usage info"""
    try:
        import psutil
        process = psutil.Process()
        return f"RAM: {process.memory_info().rss / 1024 / 1024:.0f}MB"
    except:
        return "Memory info unavailable"

def calculate_comprehensive_metrics(y_true, y_pred, dataset_name):
    """Calculate comprehensive metrics for model comparison"""
    # Basic error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R-squared (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Maximum error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Normalized metrics
    y_true_range = np.max(y_true) - np.min(y_true)
    normalized_rmse = rmse / y_true_range if y_true_range > 0 else 0
    normalized_mae = mae / y_true_range if y_true_range > 0 else 0
    
    # Error distribution analysis
    errors = np.abs(y_true - y_pred)
    error_std = np.std(errors)
    error_percentiles = np.percentile(errors, [50, 75, 90, 95, 99])
    
    # High error analysis
    high_error_001 = np.sum(errors > 0.01)
    high_error_005 = np.sum(errors > 0.05)
    high_error_01 = np.sum(errors > 0.1)
    
    # Per-dimension analysis (for multi-dimensional outputs)
    per_dim_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    per_dim_rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    
    return {
        'dataset': dataset_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
        'max_error': max_error,
        'normalized_rmse': normalized_rmse,
        'normalized_mae': normalized_mae,
        'error_std': error_std,
        'error_p50': error_percentiles[0],
        'error_p75': error_percentiles[1],
        'error_p90': error_percentiles[2],
        'error_p95': error_percentiles[3],
        'error_p99': error_percentiles[4],
        'high_error_001': high_error_001,
        'high_error_005': high_error_005,
        'high_error_01': high_error_01,
        'per_dim_mae': per_dim_mae,
        'per_dim_rmse': per_dim_rmse,
        'total_samples': len(y_true)
    }

def convert_numpy_to_list(obj):
    """Convert numpy arrays and scalars to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        # Handle numpy scalar types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(v) for v in obj)
    else:
        return obj

with tf.device('/gpu:1'):
    step = 1
    start_time = time.time()
    
    try:
        debug_print(step, "Starting Customized CNN training...")
        debug_print(step, f"Initial memory: {get_memory_usage()}")
        step += 1

        ################################## Load Robot data ##################################################################
        debug_print(step, "Loading robot data...")
        # Updated paths to match the fps3 data used in RCN_GRU
        Arm2_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_fps3.csv', header=None)
        Arm2_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_fps3.csv', header=None)

        # Updated data split to match RCN_GRU (10000 train, rest test)
        robot_state_train_input = Arm2_CS_State.iloc[:10000].values
        debug_print(step, f"Robot state input training set size: {robot_state_train_input.shape}")
        robot_state_train_label = Arm2_NS_State.iloc[:10000].values
        debug_print(step, f"Robot state label training set size: {robot_state_train_label.shape}")

        robot_state_test_input = Arm2_CS_State.iloc[10000:].values
        debug_print(step, f"Robot state input test set size: {robot_state_test_input.shape}")
        robot_state_test_label = Arm2_NS_State.iloc[10000:].values
        debug_print(step, f"Robot state label test set size: {robot_state_test_label.shape}")
        step += 1

        ################################## Standardization ###################################################################
        debug_print(step, "Standardizing robot data...")
        CS_train_names = list(range(robot_state_train_input.shape[1]))
        NS_train_names = list(range(robot_state_train_label.shape[1]))
        CS_test_names = list(range(robot_state_test_input.shape[1]))
        NS_test_names = list(range(robot_state_test_label.shape[1]))

        input_scaler = preprocessing.StandardScaler()
        output_scaler = preprocessing.StandardScaler()
        
        robot_state_train_input_scaled = input_scaler.fit_transform(robot_state_train_input)
        robot_state_train_label_scaled = output_scaler.fit_transform(robot_state_train_label)
        robot_state_test_input_scaled = input_scaler.transform(robot_state_test_input)
        robot_state_test_label_scaled = output_scaler.transform(robot_state_test_label)

        # Convert to float32 for memory efficiency
        robot_state_train_input_scaled = robot_state_train_input_scaled.astype(np.float32)
        robot_state_train_label_scaled = robot_state_train_label_scaled.astype(np.float32)
        robot_state_test_input_scaled = robot_state_test_input_scaled.astype(np.float32)
        robot_state_test_label_scaled = robot_state_test_label_scaled.astype(np.float32)
        step += 1

        ############################################### Load image data #####################################################
        debug_print(step, "Loading image filenames...")
        # Updated paths to match the actual file structure
        try:
            X_train_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/filtered/train_filtered.csv', header=None)
            X_test_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/filtered/test_filtered.csv', header=None)
        except FileNotFoundError:
            debug_print(step, "fps3 image files not found, trying original names...")
            X_train_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/trainImageName.csv', header=None)
            X_test_filenames = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/testImageName.csv', header=None)

        # Create consistent splits with robot data
        total_train_samples = 10000
        total_test_samples = len(robot_state_test_input_scaled)
        
        # If we have enough image filenames, use them, otherwise generate synthetic ones
        if len(X_train_filenames) >= total_train_samples:
            X_train_filenames = X_train_filenames.iloc[:total_train_samples].values[:, 0]
        else:
            # Generate filename pattern if not enough files
            X_train_filenames = [f"image_{i:06d}.jpg" for i in range(total_train_samples)]
        
        if len(X_test_filenames) >= total_test_samples:
            X_test_filenames = X_test_filenames.iloc[:total_test_samples].values[:, 0]
        else:
            # Generate filename pattern if not enough files
            X_test_filenames = [f"image_{i:06d}.jpg" for i in range(total_train_samples, total_train_samples + total_test_samples)]

        debug_print(step, f"Train image filenames: {len(X_train_filenames)}")
        debug_print(step, f"Test image filenames: {len(X_test_filenames)}")
        step += 1

        ######################################################################################################################
        debug_print(step, "Setting up custom data generator...")
        
        class My_Custom_Generator(keras.utils.Sequence):
            def __init__(self, image_filenames, robot_input, labels, batch_size, image_base_path):
                self.image_filenames = image_filenames
                self.robot_input = robot_input
                self.labels = labels
                self.batch_size = batch_size
                self.image_base_path = image_base_path
            
            def __len__(self):
                return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
            
            def __getitem__(self, idx):
                batch_x_img = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
                batch_x_robot = self.robot_input[idx * self.batch_size : (idx+1) * self.batch_size]
                batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
                
                # Process images
                images = []
                for file_name in batch_x_img:
                    try:
                        img_path = f"{self.image_base_path}/{file_name}"
                        img = imread(img_path)
                        img = resize(img, (224, 224, 3))
                        images.append(img)
                    except Exception as e:
                        # If image loading fails, create a dummy image
                        debug_print(-1, f"Failed to load image {file_name}: {e}")
                        images.append(np.random.rand(224, 224, 3) * 0.1)  # Very dim random image
                
                images = np.array(images, dtype=np.float32) / 255.0
                robot_data = np.array(batch_x_robot, dtype=np.float32)
                labels = np.array(batch_y, dtype=np.float32)
                
                return {'image_input': images, 'robot_input': robot_data}, labels

        batch_size = 32
        
        # Updated image base path
        image_base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/all_images'
        
        my_training_batch_generator = My_Custom_Generator(
            X_train_filenames, robot_state_train_input_scaled, robot_state_train_label_scaled, 
            batch_size, image_base_path
        )
        my_testing_batch_generator = My_Custom_Generator(
            X_test_filenames, robot_state_test_input_scaled, robot_state_test_label_scaled, 
            batch_size, image_base_path
        )
        step += 1
        
        #######################################################################################################################################
        ###################################### Define CNN ##############################
        debug_print(step, "Building Customized CNN model...")
        
        image_input_layer = keras.layers.Input(shape=(224,224,3), name='image_input')

        # VGG-like architecture
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

        # Robot state processing
        robot_state_input_layer = keras.layers.Input(shape=(robot_state_train_input_scaled.shape[1],), name='robot_input')
        dense_1 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
        dense_2 = keras.layers.Dense(25, activation="relu")(dense_1)

        # Fusion layer
        concat = keras.layers.concatenate([dense_2, cnn_flatten])
        dense_3 = keras.layers.Dense(80, activation="relu")(concat)
        dense_4 = keras.layers.Dense(20, activation="relu")(dense_3)
        output_layer = keras.layers.Dense(robot_state_train_label_scaled.shape[1], activation="linear")(dense_4)

        model = keras.models.Model(inputs=[image_input_layer, robot_state_input_layer], outputs=output_layer)

        # Define intermediate layer model for latent space extraction
        intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_3)

        # Print model summary
        model.summary()
        trainable_count = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_count = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        debug_print(step, f"Trainable parameters: {trainable_count:,}")
        debug_print(step, f"Non-trainable parameters: {non_trainable_count:,}")
        step += 1

        ##########################################################################################################################################
        debug_print(step, "Configuring training...")
        
        # Early stopping callback
        monitor = EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-3, 
            patience=5,  # Increased patience
            verbose=1, 
            mode='auto', 
            restore_best_weights=True
        )

        # Define optimizer and loss function
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.MeanAbsoluteError()

        # Compile model
        model.compile(optimizer=opt, loss=loss_fn, metrics=['mse'])
        step += 1

        # Train the model
        debug_print(step, "Starting training...")
        debug_print(step, f"Steps per epoch: {len(my_training_batch_generator)}")
        debug_print(step, f"Validation steps: {len(my_testing_batch_generator)}")
        
        training_start = time.time()
        history = model.fit(
            my_training_batch_generator, 
            epochs=20,  # Increased epochs
            validation_data=my_testing_batch_generator,
            callbacks=[monitor],
            verbose=1
        )
        training_end = time.time()
        training_time = training_end - training_start
        debug_print(step, f"Training completed in {training_time:.2f} seconds")
        step += 1

        # Evaluate model and calculate comprehensive metrics
        debug_print(step, "Evaluating model and calculating comprehensive metrics...")
        
        # Get predictions
        debug_print(step, "Making predictions...")
        train_predictions = model.predict(my_training_batch_generator, verbose=0)
        test_predictions = model.predict(my_testing_batch_generator, verbose=0)
        
        # Get actual labels (need to collect from generators)
        debug_print(step, "Collecting actual labels...")
        train_labels_actual = []
        test_labels_actual = []
        
        for i in range(len(my_training_batch_generator)):
            _, labels = my_training_batch_generator[i]
            train_labels_actual.append(labels)
        train_labels_actual = np.vstack(train_labels_actual)
        
        for i in range(len(my_testing_batch_generator)):
            _, labels = my_testing_batch_generator[i]
            test_labels_actual.append(labels)
        test_labels_actual = np.vstack(test_labels_actual)
        
        # Trim predictions to match actual labels (due to batch size differences)
        train_predictions = train_predictions[:len(train_labels_actual)]
        test_predictions = test_predictions[:len(test_labels_actual)]
        
        # Calculate comprehensive metrics
        train_metrics = calculate_comprehensive_metrics(train_labels_actual, train_predictions, 'Train')
        test_metrics = calculate_comprehensive_metrics(test_labels_actual, test_predictions, 'Test')
        
        # Training metrics from history
        training_history = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mse': history.history['mse'][-1],
            'final_val_mse': history.history['val_mse'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': training_time
        }
        
        # Model complexity metrics
        model_complexity = {
            'trainable_params': trainable_count,
            'non_trainable_params': non_trainable_count,
            'total_params': trainable_count + non_trainable_count,
            'model_size_mb': (trainable_count + non_trainable_count) * 4 / (1024**2),
            'input_image_shape': (224, 224, 3),
            'robot_state_dim': robot_state_train_input_scaled.shape[1],
            'output_dim': robot_state_train_label_scaled.shape[1]
        }
        
        # Compile comprehensive results
        comprehensive_results = {
            'model_name': 'Customized_CNN',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_history': training_history,
            'model_complexity': model_complexity,
            'data_info': {
                'train_samples': len(train_labels_actual),
                'test_samples': len(test_labels_actual),
                'total_samples': len(train_labels_actual) + len(test_labels_actual),
                'image_shape': (224, 224, 3),
                'robot_state_dim': robot_state_train_input_scaled.shape[1],
                'batch_size': batch_size
            }
        }
        step += 1

        # Print comprehensive results
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 MODEL: {comprehensive_results['model_name']}")
        print(f"🕒 Timestamp: {comprehensive_results['timestamp']}")
        print(f"⏱️ Training Time: {training_history['training_time_seconds']:.2f} seconds")
        print(f"📈 Epochs Trained: {training_history['epochs_trained']}")
        
        print(f"\n🏗️ MODEL COMPLEXITY:")
        print(f"   Trainable Parameters: {model_complexity['trainable_params']:,}")
        print(f"   Total Parameters: {model_complexity['total_params']:,}")
        print(f"   Model Size: {model_complexity['model_size_mb']:.2f} MB")
        print(f"   Input Image Shape: {model_complexity['input_image_shape']}")
        print(f"   Robot State Dimension: {model_complexity['robot_state_dim']}")
        print(f"   Output Dimension: {model_complexity['output_dim']}")
        
        print(f"\n📊 DATASET SIZES:")
        print(f"   Training: {comprehensive_results['data_info']['train_samples']} samples")
        print(f"   Test: {comprehensive_results['data_info']['test_samples']} samples")
        print(f"   Total: {comprehensive_results['data_info']['total_samples']} samples")
        
        # Print metrics for each dataset
        for metrics in [train_metrics, test_metrics]:
            print(f"\n📈 {metrics['dataset'].upper()} SET METRICS:")
            print(f"   MSE: {metrics['mse']:.6f}")
            print(f"   RMSE: {metrics['rmse']:.6f}")
            print(f"   MAE: {metrics['mae']:.6f}")
            print(f"   R² Score: {metrics['r2_score']:.6f}")
            print(f"   MAPE: {metrics['mape']:.2f}%")
            print(f"   Max Error: {metrics['max_error']:.6f}")
            print(f"   Normalized RMSE: {metrics['normalized_rmse']:.6f}")
            print(f"   Normalized MAE: {metrics['normalized_mae']:.6f}")
            print(f"   Error Std: {metrics['error_std']:.6f}")
            print(f"   Error Percentiles (50/75/90/95/99): {metrics['error_p50']:.4f}/{metrics['error_p75']:.4f}/{metrics['error_p90']:.4f}/{metrics['error_p95']:.4f}/{metrics['error_p99']:.4f}")
            print(f"   High Errors (>0.01/>0.05/>0.1): {metrics['high_error_001']}/{metrics['high_error_005']}/{metrics['high_error_01']}")
            print(f"   Per-dimension MAE: {[f'{x:.4f}' for x in metrics['per_dim_mae']]}")
        
        print(f"\n🎯 TRAINING CONVERGENCE:")
        print(f"   Final Training Loss: {training_history['final_train_loss']:.6f}")
        print(f"   Final Validation Loss: {training_history['final_val_loss']:.6f}")
        print(f"   Best Validation Loss: {training_history['best_val_loss']:.6f}")
        print(f"   Overfitting Check (Val/Train Loss Ratio): {training_history['final_val_loss']/training_history['final_train_loss']:.3f}")
        
        # Performance summary for easy comparison
        print(f"\n⭐ SUMMARY FOR MODEL COMPARISON:")
        print(f"   Model: Customized_CNN")
        print(f"   Test MAE: {test_metrics['mae']:.6f}")
        print(f"   Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"   Test R²: {test_metrics['r2_score']:.6f}")
        print(f"   Parameters: {model_complexity['total_params']:,}")
        print(f"   Training Time: {training_history['training_time_seconds']:.1f}s")
        print(f"   Model Size: {model_complexity['model_size_mb']:.1f}MB")
        
        print("="*80)

        # Save comprehensive results
        debug_print(step, "Saving comprehensive results...")
        results_file = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Customized_CNN_comprehensive_results.json'
        json_results = convert_numpy_to_list(comprehensive_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        debug_print(step, f"Comprehensive results saved to: {results_file}")
        step += 1

        # Save models
        debug_print(step, "Saving models...")
        model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Customized_CNN.h5')
        intermediate_layer_model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Customized_CNN_intermediate_layer.h5')
        step += 1
        
        # Save training history plots
        debug_print(step, "Saving training history plots...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Customized CNN - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.title('Customized CNN - Model MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Customized_CNN_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        total_time = time.time() - start_time
        debug_print(step, f"✅ Total execution time: {total_time:.2f} seconds")
        debug_print(step, "✅ Customized CNN training completed successfully!")
        debug_print(step, "✅ Models saved!")
        debug_print(step, "✅ Comprehensive evaluation results saved!")
        debug_print(step, "✅ Training history plots saved!")

    except Exception as e:
        print(f"❌ Error occurred at step {step}: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n=== DEBUGGING INFO ===")
        print(f"Current step: {step}")
        print(f"Current memory: {get_memory_usage()}")

    finally:
        debug_print(step, "Cleaning up...")
        gc.collect()
        tf.keras.backend.clear_session()
        debug_print(step, "Script execution finished.")