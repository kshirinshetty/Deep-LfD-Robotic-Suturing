# pyright: reportMissingImports=false
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        epochs=40,
        callbacks=[early_stop],
        validation_data=test_gen,
        verbose=1
    )
    end = time.time()
    training_time = end - start
    print(f"\n⏱️ Training took {training_time:.2f} seconds.")

    # Save the model
    model.save('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/AlexNet.h5')
    print("✅ Model training completed and saved successfully!")
    
    # ========== COMPREHENSIVE STATISTICS SECTION ==========
    print("\n" + "="*80)
    print("🔍 CALCULATING COMPREHENSIVE STATISTICS")
    print("="*80)
    
    # Get predictions
    print("Making predictions...")
    train_predictions = model.predict(train_gen, verbose=0)
    test_predictions = model.predict(test_gen, verbose=0)
    
    # Get actual labels (need to collect from generators)
    print("Collecting actual labels...")
    train_labels_actual = []
    test_labels_actual = []
    
    for i in range(len(train_gen)):
        _, labels = train_gen[i]
        train_labels_actual.append(labels)
    train_labels_actual = np.vstack(train_labels_actual)
    
    for i in range(len(test_gen)):
        _, labels = test_gen[i]
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
        'robot_state_dim': robot_state_train_input.shape[1],
        'output_dim': robot_state_train_label.shape[1]
    }
    
    # Compile comprehensive results
    comprehensive_results = {
        'model_name': 'AlexNet',
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
            'robot_state_dim': robot_state_train_input.shape[1],
            'batch_size': batch_size
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
    print(f"   Model: AlexNet")
    print(f"   Test MAE: {test_metrics['mae']:.6f}")
    print(f"   Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"   Test R²: {test_metrics['r2_score']:.6f}")
    print(f"   Parameters: {model_complexity['total_params']:,}")
    print(f"   Training Time: {training_history['training_time_seconds']:.1f}s")
    print(f"   Model Size: {model_complexity['model_size_mb']:.1f}MB")
    
    print("="*80)
    
    # Save comprehensive results
    print("Saving comprehensive results...")
    results_file = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/AlexNet_comprehensive_results.json'
    json_results = convert_numpy_to_list(comprehensive_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Comprehensive results saved to: {results_file}")
    
    # Save training history plots
    print("Saving training history plots...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('AlexNet - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('AlexNet - Model MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/AlexNet_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive evaluation completed!")
    print("✅ Training history plots saved!")
    print("✅ All statistics saved to JSON file!")
