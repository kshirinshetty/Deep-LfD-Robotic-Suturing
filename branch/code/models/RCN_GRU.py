# pyright: reportMissingImports=false
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import time
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn import preprocessing

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

with tf.device('/gpu:0'):
    step = 1
    try:
        debug_print(step, "Starting data loading...")
        debug_print(step, f"Initial memory: {get_memory_usage()}")
        step += 1
        
        # Load robot state data
        Arm2_CS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_CS_fps3.csv',header=None)
        Arm2_NS_State = pd.read_csv('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/Arm2_NS_fps3.csv',header=None)
        
        debug_print(step, f"Robot CS data shape: {Arm2_CS_State.shape}")
        debug_print(step, f"Robot NS data shape: {Arm2_NS_State.shape}")
        step += 1
        
        # Load image data with memory mapping (doesn't load into RAM immediately)
        debug_print(step, "Loading image data with memory mapping...")
        X = np.load('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/Test1-60_fps3_images.npy', mmap_mode='r')
        debug_print(step, f"Image data shape: {X.shape}")
        debug_print(step, f"Image data dtype: {X.dtype}")
        step += 1
        
        # Don't convert entire array at once - we'll do it in chunks
        debug_print(step, "Skipping full array conversion - will process in chunks")
        debug_print(step, f"Memory after loading: {get_memory_usage()}")
        step += 1

        # Prepare robot state data first (small memory footprint)
        debug_print(step, "Preparing robot state data...")
        robot_state_train_input = Arm2_CS_State.iloc[:10000].values.astype(np.float32)
        robot_state_train_label = Arm2_NS_State.iloc[:10000].values.astype(np.float32)
        robot_state_test_input = Arm2_CS_State.iloc[10000:].values.astype(np.float32)
        robot_state_test_label = Arm2_NS_State.iloc[10000:].values.astype(np.float32)

        debug_print(step, f"Robot train input shape: {robot_state_train_input.shape}")
        debug_print(step, f"Robot train label shape: {robot_state_train_label.shape}")
        debug_print(step, f"Robot test input shape: {robot_state_test_input.shape}")
        debug_print(step, f"Robot test label shape: {robot_state_test_label.shape}")
        step += 1

        # Standardize robot data
        debug_print(step, "Standardizing robot data...")
        input_scaler = preprocessing.StandardScaler().fit(robot_state_train_input)
        output_scaler = preprocessing.StandardScaler().fit(robot_state_train_label)

        robot_state_train_input = input_scaler.transform(robot_state_train_input)
        robot_state_train_label = output_scaler.transform(robot_state_train_label)
        robot_state_test_input = input_scaler.transform(robot_state_test_input)
        robot_state_test_label = output_scaler.transform(robot_state_test_label)
        step += 1

        # Load intermediate model
        debug_print(step, "Loading intermediate model...")
        intermediate_model_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/CNN_intermediate_layer.h5'
        
        try:
            intermediate_layer_model = load_model(intermediate_model_path)
            debug_print(step, "Intermediate model loaded successfully")
        except Exception as e:
            debug_print(step, f"Intermediate model not found: {e}")
            debug_print(step, "Creating from AlexNet...")
            alexnet_model = load_model('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/AlexNet.h5')
            
            intermediate_layer_model = keras.Model(
                inputs=alexnet_model.inputs,
                outputs=alexnet_model.layers[-3].output  # Get output from Dense(80) layer
            )
            
            intermediate_layer_model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/CNN_intermediate_layer.h5')
            debug_print(step, "Intermediate model created and saved from AlexNet")
        step += 1

        # Process images in small batches to avoid memory issues
        debug_print(step, "Processing images in batches...")
        batch_size = 32  # Process 32 images at a time
        
        # Initialize lists to store results
        train_intermediate_outputs = []
        test_intermediate_outputs = []
        
        # Process training data (first 10,000 images)
        debug_print(step, "Processing training images...")
        for i in range(0, 10000, batch_size):
            end_idx = min(i + batch_size, 10000)
            
            # Load and convert batch
            batch_images = X[i:end_idx].astype(np.float32)
            
            # Normalize to [0,1] if needed
            if np.max(batch_images) > 1.0:
                batch_images = batch_images / 255.0
            
            # Get corresponding robot state data
            batch_robot_state = robot_state_train_input[i:end_idx]
            
            # Predict
            batch_output = intermediate_layer_model.predict(
                [batch_images, batch_robot_state], 
                batch_size=batch_size, 
                verbose=0
            )
            
            train_intermediate_outputs.append(batch_output)
            
            # Clean up batch from memory
            del batch_images, batch_robot_state, batch_output
            gc.collect()
            
            if i % (batch_size * 10) == 0:
                debug_print(step, f"Processed {i}/10000 training images. Memory: {get_memory_usage()}")
        
        # Combine training outputs
        intermediate_output_train = np.vstack(train_intermediate_outputs)
        del train_intermediate_outputs
        gc.collect()
        
        debug_print(step, f"Training intermediate output shape: {intermediate_output_train.shape}")
        debug_print(step, f"Memory after training processing: {get_memory_usage()}")
        step += 1
        
        # Process test data (remaining 3,560 images)
        debug_print(step, "Processing test images...")
        for i in range(10000, X.shape[0], batch_size):
            end_idx = min(i + batch_size, X.shape[0])
            batch_idx = i - 10000  # Adjust index for test arrays
            
            # Load and convert batch
            batch_images = X[i:end_idx].astype(np.float32)
            
            # Normalize to [0,1] if needed
            if np.max(batch_images) > 1.0:
                batch_images = batch_images / 255.0
            
            # Get corresponding robot state data
            test_end_idx = min(batch_idx + batch_size, len(robot_state_test_input))
            batch_robot_state = robot_state_test_input[batch_idx:test_end_idx]
            
            # Predict
            batch_output = intermediate_layer_model.predict(
                [batch_images, batch_robot_state], 
                batch_size=batch_size, 
                verbose=0
            )
            
            test_intermediate_outputs.append(batch_output)
            
            # Clean up batch from memory
            del batch_images, batch_robot_state, batch_output
            gc.collect()
            
            if (i - 10000) % (batch_size * 10) == 0:
                debug_print(step, f"Processed {i-10000}/{X.shape[0]-10000} test images. Memory: {get_memory_usage()}")
        
        # Combine test outputs
        intermediate_output_test = np.vstack(test_intermediate_outputs)
        del test_intermediate_outputs
        gc.collect()
        
        debug_print(step, f"Test intermediate output shape: {intermediate_output_test.shape}")
        debug_print(step, f"Memory after test processing: {get_memory_usage()}")
        step += 1

        # Clear the original image array from memory
        del X
        gc.collect()
        debug_print(step, f"Memory after clearing image array: {get_memory_usage()}")
        step += 1

        # Time series windowing
        debug_print(step, "Setting up time series parameters...")
        validation_set_size = min(1500, len(intermediate_output_train) // 5)
        timeWindow = 4

        debug_print(step, f"Using validation set size: {validation_set_size}")
        debug_print(step, f"Time window: {timeWindow}")
        step += 1

        # Split intermediate outputs
        debug_print(step, "Splitting intermediate outputs...")
        train_set = intermediate_output_train[:-validation_set_size]
        validation_set = intermediate_output_train[-validation_set_size:]
        test_set = intermediate_output_test

        debug_print(step, f"Train set shape: {train_set.shape}")
        debug_print(step, f"Validation set shape: {validation_set.shape}")
        debug_print(step, f"Test set shape: {test_set.shape}")
        step += 1

        def create_timeseries_matrix(data, output=None):
            if data.shape[0] <= timeWindow:
                raise ValueError(f"Not enough data for time series. Need > {timeWindow}, got {data.shape[0]}")
            
            samples = data.shape[0] - timeWindow
            if samples <= 0:
                raise ValueError(f"Invalid samples count: {samples}")
                
            X_out = np.zeros((samples, timeWindow, data.shape[1]), dtype=np.float32)
            for i in range(samples):
                X_out[i] = data[i:i+timeWindow]
            
            if output is not None:
                if len(output) < timeWindow:
                    raise ValueError(f"Output data too short. Need >= {timeWindow}, got {len(output)}")
                Y_out = output[timeWindow:timeWindow+samples]
                return X_out, Y_out
            return X_out

        # Create time series matrices
        debug_print(step, "Creating time series matrices...")
        
        # For training set
        train_labels = robot_state_train_label[:-validation_set_size]
        train_matrix, output_train_matrix = create_timeseries_matrix(train_set, train_labels)
        
        # For validation set
        validation_labels = robot_state_train_label[-validation_set_size:]
        validation_matrix, output_validation_matrix = create_timeseries_matrix(validation_set, validation_labels)
        
        # For test set
        test_matrix, output_test_matrix = create_timeseries_matrix(test_set, robot_state_test_label)

        debug_print(step, f"Train matrix shape: {train_matrix.shape}")
        debug_print(step, f"Train output shape: {output_train_matrix.shape}")
        debug_print(step, f"Validation matrix shape: {validation_matrix.shape}")
        debug_print(step, f"Validation output shape: {output_validation_matrix.shape}")
        debug_print(step, f"Test matrix shape: {test_matrix.shape}")
        debug_print(step, f"Test output shape: {output_test_matrix.shape}")
        step += 1

        # Clean up intermediate outputs from memory
        del intermediate_output_train, intermediate_output_test, train_set, validation_set, test_set
        gc.collect()
        debug_print(step, f"Memory after cleanup: {get_memory_usage()}")
        step += 1

        debug_print(step, "Building GRU model...")
        feature_dim = train_matrix.shape[2]  # Get feature dimension from time series matrix
        output_dim = robot_state_train_label.shape[1]
        
        debug_print(step, f"Feature dimension: {feature_dim}")
        debug_print(step, f"Output dimension: {output_dim}")
        
        GRU_Model = keras.models.Sequential([
            keras.layers.GRU(150, return_sequences=True, input_shape=(timeWindow, feature_dim)),
            keras.layers.Dropout(0.2),
            keras.layers.GRU(100),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(output_dim, activation='linear')
        ])

        GRU_Model.compile(loss='mae', optimizer=keras.optimizers.Adam(1e-3), metrics=['mse'])
        
        # Print model summary
        GRU_Model.summary()
        trainable_count = np.sum([np.prod(v.shape) for v in GRU_Model.trainable_weights])
        non_trainable_count = np.sum([np.prod(v.shape) for v in GRU_Model.non_trainable_weights])
        debug_print(step, f"Trainable parameters: {trainable_count:,}")
        debug_print(step, f"Non-trainable parameters: {non_trainable_count:,}")
        step += 1

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, restore_best_weights=True)

        debug_print(step, "Starting training...")
        debug_print(step, f"Memory before training: {get_memory_usage()}")
        start = time.time()
        history = GRU_Model.fit(
            train_matrix, output_train_matrix,
            validation_data=(validation_matrix, output_validation_matrix),
            callbacks=[monitor],
            epochs=40,
            batch_size=32,
            verbose=1
        )
        end = time.time()
        debug_print(step, f"Training took {(end - start):.2f} seconds.")
        step += 1

        debug_print(step, "Evaluating...")
        loss, mse = GRU_Model.evaluate(test_matrix, output_test_matrix, verbose=0)
        debug_print(step, f"Test loss: {loss:.4f}, Test MSE: {mse:.4f}")
        step += 1

        debug_print(step, "Predicting and calculating comprehensive metrics...")
        
        # Make predictions
        test_predict_gru = GRU_Model.predict(test_matrix, batch_size=32)
        train_predict_gru = GRU_Model.predict(train_matrix, batch_size=32)
        val_predict_gru = GRU_Model.predict(validation_matrix, batch_size=32)
        
        # Calculate comprehensive metrics
        def calculate_metrics(y_true, y_pred, dataset_name):
            """Calculate comprehensive metrics for model comparison"""
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
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
        
        # Calculate metrics for all datasets
        test_metrics = calculate_metrics(output_test_matrix, test_predict_gru, 'Test')
        train_metrics = calculate_metrics(output_train_matrix, train_predict_gru, 'Train')
        val_metrics = calculate_metrics(output_validation_matrix, val_predict_gru, 'Validation')
        
        # Training metrics from history
        training_history = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mse': history.history['mse'][-1],
            'final_val_mse': history.history['val_mse'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': end - start
        }
        
        # Model complexity metrics
        model_complexity = {
            'trainable_params': trainable_count,
            'non_trainable_params': non_trainable_count,
            'total_params': trainable_count + non_trainable_count,
            'model_size_mb': (trainable_count + non_trainable_count) * 4 / (1024**2),  # Assuming float32
            'time_window': timeWindow,
            'feature_dim': feature_dim,
            'output_dim': output_dim
        }
        
        # Compile comprehensive results
        comprehensive_results = {
            'model_name': 'RCN_GRU',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'training_history': training_history,
            'model_complexity': model_complexity,
            'data_info': {
                'train_samples': len(train_matrix),
                'val_samples': len(validation_matrix),
                'test_samples': len(test_matrix),
                'total_original_samples': 13560,
                'image_shape': (224, 224, 3),
                'robot_state_dim': robot_state_train_input.shape[1]
            }
        }
        
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
        print(f"   Time Window: {model_complexity['time_window']}")
        print(f"   Feature Dimension: {model_complexity['feature_dim']}")
        print(f"   Output Dimension: {model_complexity['output_dim']}")
        
        print(f"\n📊 DATASET SIZES:")
        print(f"   Training: {comprehensive_results['data_info']['train_samples']} samples")
        print(f"   Validation: {comprehensive_results['data_info']['val_samples']} samples")
        print(f"   Test: {comprehensive_results['data_info']['test_samples']} samples")
        
        # Print metrics for each dataset
        for metrics in [test_metrics, train_metrics, val_metrics]:
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
        print(f"   Model: RCN_GRU")
        print(f"   Test MAE: {test_metrics['mae']:.6f}")
        print(f"   Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"   Test R²: {test_metrics['r2_score']:.6f}")
        print(f"   Parameters: {model_complexity['total_params']:,}")
        print(f"   Training Time: {training_history['training_time_seconds']:.1f}s")
        print(f"   Model Size: {model_complexity['model_size_mb']:.1f}MB")
        
        print("="*80)
        
        # Save comprehensive results to file
        import json
        results_file = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/RCN_GRU_comprehensive_results.json'
        # Replace the convert_numpy_to_list function with this improved version:

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
        
        json_results = convert_numpy_to_list(comprehensive_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        debug_print(step, f"Comprehensive results saved to: {results_file}")
        step += 1

        debug_print(step, "Saving model...")
        GRU_Model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/RCN_GRU.h5')
        
        # Save training history plots
        debug_print(step, "Saving training history plots...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.title('Model MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/RCN_GRU_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        debug_print(step, "✅ Model training completed and saved successfully!")
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