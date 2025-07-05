"""
Training module for Brain Tumor CNN Classification
Handles model training, validation, and monitoring
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from config import config
from model import BrainTumorDenseNet
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self, model_type='densenet', densenet_variant='DenseNet121'):
        self.config = config
        self.model_type = model_type
        self.densenet_variant = densenet_variant
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor()
        
        # Create directories
        self.config.create_directories()
        
    def setup_model(self):
        """Initialize and build the DenseNet model"""
        densenet_model = BrainTumorDenseNet()
        self.model = densenet_model.build_model(
            model_type=self.model_type,
            densenet_variant=self.densenet_variant
        )
        
        print(f"Model setup complete: {self.model_type}")
        if self.model_type == 'densenet_transfer':
            print(f"DenseNet variant: {self.densenet_variant}")
        
        return self.model
    
    def setup_callbacks(self, model_name=None):
        """Setup training callbacks"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"
        
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        checkpoint_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{model_name}_best.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LR,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_path = os.path.join(self.config.LOGS_PATH, model_name)
        tensorboard = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_path = os.path.join(self.config.LOGS_PATH, f"{model_name}_training.csv")
        csv_logger = CSVLogger(csv_path, append=True)
        callbacks.append(csv_logger)
        
        return callbacks, model_name
    
    def prepare_data(self):
        """Prepare training, validation, and test data"""
        print("Preparing data generators...")
        
        # Create data generators
        train_gen, val_gen, test_gen = self.preprocessor.create_data_generators()
        
        # Convert generators to tf.data.Dataset for better performance
        train_dataset = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.INPUT_SHAPE), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.INPUT_SHAPE), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_generator(
            test_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.INPUT_SHAPE), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
    
    def calculate_steps(self):
        """Calculate steps per epoch based on dataset size"""
        stats = self.preprocessor.get_dataset_statistics()
        total_images = stats['total_images']
        
        # Calculate approximate split sizes
        train_size = int(total_images * (1 - self.config.VALIDATION_SPLIT - self.config.TEST_SPLIT))
        val_size = int(total_images * self.config.VALIDATION_SPLIT)
        
        steps_per_epoch = max(1, train_size // self.config.BATCH_SIZE)
        validation_steps = max(1, val_size // self.config.BATCH_SIZE)
        
        print(f"Estimated steps per epoch: {steps_per_epoch}")
        print(f"Estimated validation steps: {validation_steps}")
        
        return steps_per_epoch, validation_steps
    
    def train_model(self, epochs=None, model_name=None):
        """Train the CNN model"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model type: {self.model_type}")
        
        # Setup model
        if self.model is None:
            self.setup_model()
        
        # Setup callbacks
        callbacks, model_name = self.setup_callbacks(model_name)
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data()
        
        # Calculate steps
        steps_per_epoch, validation_steps = self.calculate_steps()
        
        # Train the model
        print("Starting model training...")
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{model_name}_final.h5")
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        self.save_training_history(model_name)
        
        return self.history, model_name
    
    def save_training_history(self, model_name):
        """Save training history to JSON file"""
        if self.history is None:
            print("No training history to save")
            return
        
        history_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_history.json")
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
    
    def plot_training_history(self, model_name=None, save_plots=True):
        """Plot training history"""
        if self.history is None:
            print("No training history to plot")
            return
        
        if model_name is None:
            model_name = f"{self.model_type}_model"
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
        
        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision', color='green')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', color='orange')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall', color='purple')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', color='brown')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_training_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {plot_path}")
        
        plt.show()
    
    def get_training_summary(self):
        """Get training summary statistics"""
        if self.history is None:
            print("No training history available")
            return None
        
        history = self.history.history
        
        summary = {
            'total_epochs': len(history['loss']),
            'best_val_accuracy': max(history['val_accuracy']),
            'best_val_loss': min(history['val_loss']),
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        
        if 'precision' in history:
            summary['best_val_precision'] = max(history['val_precision'])
            summary['final_val_precision'] = history['val_precision'][-1]
        
        if 'recall' in history:
            summary['best_val_recall'] = max(history['val_recall'])
            summary['final_val_recall'] = history['val_recall'][-1]
        
        return summary

# Example usage
if __name__ == "__main__":
    print("Brain Tumor CNN Training")
    print("=" * 50)
    
    # Train custom CNN
    print("Training Custom CNN...")
    trainer = ModelTrainer(model_type='custom')
    history, model_name = trainer.train_model(epochs=10)  # Reduced epochs for testing
    trainer.plot_training_history(model_name)
    
    # Print training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
