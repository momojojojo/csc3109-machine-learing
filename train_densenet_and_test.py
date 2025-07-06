#!/usr/bin/env python3
"""
DenseNet Training and Test Evaluation Script
This script trains the DenseNet model and then evaluates it on the test dataset.
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configuration
DATASET_PATH = "C:/Users/danie/csc3109-machine-learing/dataset_8"
TEST_DATA_PATH = "C:/Users/danie/csc3109-machine-learing/test"
MODEL_PATH = "C:/Users/danie/csc3109-machine-learing/final_densenet121_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def train_model():
    """Train the DenseNet model if it doesn't exist"""
    
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists: {MODEL_PATH}")
        return load_model(MODEL_PATH)
    
    print("🚀 Starting DenseNet121 training...")
    
    # Data augmentation
    train_data_generator = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    validation_data_generator = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3 
    )
    
    # Create generators
    training_generator = train_data_generator.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    validation_generator = validation_data_generator.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"Training samples: {training_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Number of classes: {training_generator.num_classes}")
    print(f"Class indices: {training_generator.class_indices}")
    
    # Build model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model architecture:")
    model.summary()
    
    # Initial training with frozen base model
    print("\n🔥 Phase 1: Training with frozen base model...")
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        training_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Fine-tuning with unfrozen base model
    print("\n🔥 Phase 2: Fine-tuning with unfrozen base model...")
    base_model.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    history_fine = model.fit(
        training_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    print(f"\n✅ Training Complete!")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")
    
    return model

def test_model(model):
    """Test the trained model on the test dataset"""
    
    print("\n" + "="*60)
    print("🧪 TESTING DENSENET121 ON REAL TEST DATASET")
    print("="*60)
    
    # Create test data generator (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,  # Important: don't shuffle for proper evaluation
        seed=42
    )

    print(f"Test samples: {test_generator.samples}")
    print(f"Number of classes: {test_generator.num_classes}")
    print(f"Class indices: {test_generator.class_indices}")
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    print(f"Class names: {class_names}")
    
    # Count samples per class
    print("\nTest dataset distribution:")
    for class_name in class_names:
        class_path = os.path.join(TEST_DATA_PATH, class_name)
        if os.path.isdir(class_path):
            file_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  - {class_name}: {file_count} images")
    
    # Evaluate model on test data
    print("\n🔍 Evaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    print(f"\n=== TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate predictions for detailed analysis
    print("\n🔮 Generating predictions for detailed analysis...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of test samples: {len(true_classes)}")
    print(f"Number of predictions: {len(predicted_classes)}")
    
    # Calculate detailed metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    print("\n=== DETAILED METRICS ===")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Generate classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, 
                                 output_dict=True)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\n=== CONFUSION MATRIX ===")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Raw counts:")
    print(cm_df)

    # Normalized confusion matrix
    cm_normalized = confusion_matrix(true_classes, predicted_classes, normalize='true')
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    print("\nNormalized (percentages):")
    print(cm_norm_df.round(3))
    
    # Analyze prediction confidence
    max_confidences = np.max(predictions, axis=1)
    correct_mask = (true_classes == predicted_classes)
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    print("\n=== PREDICTION CONFIDENCE ANALYSIS ===")
    print(f"Average confidence for correct predictions: {np.mean(correct_confidences):.4f}")
    if len(incorrect_confidences) > 0:
        print(f"Average confidence for incorrect predictions: {np.mean(incorrect_confidences):.4f}")
    else:
        print("No incorrect predictions!")
    
    # Create comprehensive results summary
    print("\n" + "="*60)
    print("           DENSENET121 TEST EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 DATASET INFORMATION:")
    print(f"   • Test samples: {test_generator.samples}")
    print(f"   • Number of classes: {test_generator.num_classes}")
    print(f"   • Classes: {', '.join(class_names)}")
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Precision (weighted): {precision:.4f} ({precision*100:.2f}%)")
    print(f"   • Recall (weighted): {recall:.4f} ({recall*100:.2f}%)")
    print(f"   • F1-Score (weighted): {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\n📈 PER-CLASS PERFORMANCE:")
    for class_name in class_names:
        class_precision = report[class_name]['precision']
        class_recall = report[class_name]['recall']
        class_f1 = report[class_name]['f1-score']
        class_support = int(report[class_name]['support'])
        print(f"   • {class_name.capitalize()}:")
        print(f"     - Precision: {class_precision:.3f} | Recall: {class_recall:.3f} | F1: {class_f1:.3f} | Samples: {class_support}")
    
    print(f"\n🔍 PREDICTION CONFIDENCE:")
    print(f"   • Avg confidence (correct): {np.mean(correct_confidences):.4f}")
    if len(incorrect_confidences) > 0:
        print(f"   • Avg confidence (incorrect): {np.mean(incorrect_confidences):.4f}")
    
    correct_predictions = np.sum(true_classes == predicted_classes)
    incorrect_predictions = len(true_classes) - correct_predictions
    
    print(f"\n✅ PREDICTION BREAKDOWN:")
    print(f"   • Correct predictions: {correct_predictions}/{len(true_classes)} ({correct_predictions/len(true_classes)*100:.2f}%)")
    print(f"   • Incorrect predictions: {incorrect_predictions}/{len(true_classes)} ({incorrect_predictions/len(true_classes)*100:.2f}%)")
    
    print(f"\n💡 MODEL INSIGHTS:")
    if test_accuracy > 0.9:
        print(f"   • Excellent performance! The model shows high accuracy across all classes.")
    elif test_accuracy > 0.8:
        print(f"   • Good performance! The model performs well with room for minor improvements.")
    elif test_accuracy > 0.7:
        print(f"   • Moderate performance. Consider additional training or data augmentation.")
    else:
        print(f"   • Performance needs improvement. Consider model architecture changes or more data.")
    
    # Save detailed results to file
    results_summary = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names,
        'total_test_samples': int(test_generator.samples)
    }
    
    import json
    with open('densenet_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: densenet_test_results.json")
    
    return results_summary

def main():
    """Main function to train and test the model"""
    
    print("🧠 DenseNet121 Brain Tumor Classification")
    print("🎯 Training on dataset_8 and Testing on real test dataset")
    print("="*60)
    
    # Check if paths exist
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Training dataset not found: {DATASET_PATH}")
        return
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Test dataset not found: {TEST_DATA_PATH}")
        return
    
    # Train or load model
    model = train_model()
    
    # Test model
    results = test_model(model)
    
    print("\n🎉 EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
