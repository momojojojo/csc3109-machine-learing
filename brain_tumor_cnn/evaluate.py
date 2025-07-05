"""
Evaluation module for Brain Tumor CNN Classification
Handles model evaluation, metrics calculation, and visualization
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import json
from config import config
from data_preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self, model_path=None):
        self.config = config
        self.model = None
        self.model_path = model_path
        self.preprocessor = DataPreprocessor()
        self.test_generator = None
        self.predictions = None
        self.true_labels = None
        
        # Create directories
        self.config.create_directories()
        
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None:
            raise ValueError("No model path provided")
            
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def prepare_test_data(self):
        """Prepare test data for evaluation"""
        print("Preparing test data...")
        
        # Create data generators
        _, _, test_gen = self.preprocessor.create_data_generators()
        self.test_generator = test_gen
        
        # Convert to tf.data.Dataset
        test_dataset = tf.data.Dataset.from_generator(
            test_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.INPUT_SHAPE), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        return test_dataset
    
    def predict_on_test_set(self, test_dataset=None):
        """Make predictions on test set"""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return None, None
            
        if test_dataset is None:
            test_dataset = self.prepare_test_data()
        
        print("Making predictions on test set...")
        
        # Collect all test data and predictions
        all_predictions = []
        all_true_labels = []
        
        for batch_images, batch_labels in test_dataset:
            # Make predictions
            batch_predictions = self.model.predict(batch_images, verbose=0)
            
            # Store predictions and true labels
            all_predictions.append(batch_predictions)
            all_true_labels.append(batch_labels.numpy())
        
        # Concatenate all batches
        self.predictions = np.vstack(all_predictions)
        self.true_labels = np.vstack(all_true_labels)
        
        print(f"Predictions completed. Shape: {self.predictions.shape}")
        
        return self.predictions, self.true_labels
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Call predict_on_test_set() first.")
            return None
        
        # Convert one-hot encoded labels to class indices
        true_classes = np.argmax(self.true_labels, axis=1)
        pred_classes = np.argmax(self.predictions, axis=1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, pred_classes, average=None
        )
        
        # Calculate macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='weighted'
        )
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        metrics = {
            'overall': overall_metrics,
            'per_class': per_class_metrics
        }
        
        return metrics
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Call predict_on_test_set() first.")
            return None
        
        true_classes = np.argmax(self.true_labels, axis=1)
        pred_classes = np.argmax(self.predictions, axis=1)
        
        report = classification_report(
            true_classes, pred_classes,
            target_names=self.config.CLASS_NAMES,
            output_dict=True
        )
        
        # Print formatted report
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(
            true_classes, pred_classes,
            target_names=self.config.CLASS_NAMES
        ))
        
        return report
    
    def plot_confusion_matrix(self, model_name=None, save_plot=True):
        """Plot confusion matrix"""
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Call predict_on_test_set() first.")
            return
        
        true_classes = np.argmax(self.true_labels, axis=1)
        pred_classes = np.argmax(self.predictions, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.CLASS_NAMES,
                   yticklabels=self.config.CLASS_NAMES)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Calculate and display accuracy for each class
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            plt.text(len(self.config.CLASS_NAMES) + 0.5, i + 0.5, 
                    f'Acc: {acc:.3f}', ha='center', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            if model_name is None:
                model_name = "model"
            plot_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_confusion_matrix.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {plot_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, model_name=None, save_plot=True):
        """Plot ROC curves for multi-class classification"""
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Call predict_on_test_set() first.")
            return
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.config.NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(self.true_labels[:, i], self.predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(
            self.true_labels.ravel(), self.predictions.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        # Colors for different classes
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        
        for i, color in zip(range(self.config.NUM_CLASSES), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.config.CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            if model_name is None:
                model_name = "model"
            plot_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_roc_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {plot_path}")
        
        plt.show()
        
        return roc_auc
    
    def plot_prediction_samples(self, num_samples=12, model_name=None, save_plot=True):
        """Plot sample predictions with confidence scores"""
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Call predict_on_test_set() first.")
            return
        
        # Get random sample indices
        sample_indices = np.random.choice(len(self.predictions), num_samples, replace=False)
        
        # Create subplot grid
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle('Sample Predictions with Confidence Scores', fontsize=16, fontweight='bold')
        
        # Load some test images for visualization
        df = self.preprocessor.load_dataset_info()
        test_df = df.sample(n=num_samples)
        
        for idx, (ax, (_, row)) in enumerate(zip(axes.flat, test_df.iterrows())):
            if idx >= num_samples:
                break
                
            # Load and preprocess image
            image = self.preprocessor.preprocess_image(row['image_path'])
            
            if image is not None:
                # Get prediction for this sample
                pred_idx = sample_indices[idx] if idx < len(sample_indices) else idx
                if pred_idx < len(self.predictions):
                    pred_probs = self.predictions[pred_idx]
                    true_label = np.argmax(self.true_labels[pred_idx])
                    pred_label = np.argmax(pred_probs)
                    confidence = np.max(pred_probs)
                    
                    # Display image
                    ax.imshow(image)
                    
                    # Create title with prediction info
                    true_class = self.config.CLASS_NAMES[true_label]
                    pred_class = self.config.CLASS_NAMES[pred_label]
                    
                    color = 'green' if true_label == pred_label else 'red'
                    title = f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}'
                    
                    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
                    ax.axis('off')
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            axes.flat[idx].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            if model_name is None:
                model_name = "model"
            plot_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_prediction_samples.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Prediction samples saved to: {plot_path}")
        
        plt.show()
    
    def save_evaluation_results(self, model_name=None):
        """Save evaluation results to JSON file"""
        if model_name is None:
            model_name = "model_evaluation"
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        classification_rep = self.generate_classification_report()
        
        # Combine results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': classification_rep,
            'config': {
                'num_classes': self.config.NUM_CLASSES,
                'class_names': self.config.CLASS_NAMES,
                'input_shape': self.config.INPUT_SHAPE
            }
        }
        
        # Save to JSON
        results_path = os.path.join(self.config.RESULTS_PATH, f"{model_name}_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to: {results_path}")
        
        return results
    
    def comprehensive_evaluation(self, model_path, model_name=None):
        """Perform comprehensive model evaluation"""
        print("Starting comprehensive model evaluation...")
        print("=" * 60)
        
        # Load model
        self.load_model(model_path)
        
        # Prepare test data and make predictions
        test_dataset = self.prepare_test_data()
        self.predict_on_test_set(test_dataset)
        
        # Generate all evaluation outputs
        if model_name is None:
            model_name = os.path.basename(model_path).replace('.h5', '')
        
        print(f"Evaluating model: {model_name}")
        
        # Calculate and display metrics
        metrics = self.calculate_metrics()
        if metrics:
            print("\nOverall Metrics:")
            for key, value in metrics['overall'].items():
                print(f"{key}: {value:.4f}")
        
        # Generate classification report
        self.generate_classification_report()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(model_name)
        self.plot_roc_curves(model_name)
        self.plot_prediction_samples(model_name=model_name)
        
        # Save results
        results = self.save_evaluation_results(model_name)
        
        print("\nComprehensive evaluation completed!")
        print("=" * 60)
        
        return results

# Example usage
if __name__ == "__main__":
    # Example evaluation
    model_path = "models/custom_20240704_123456_best.h5"  # Replace with actual model path
    
    if os.path.exists(model_path):
        evaluator = ModelEvaluator()
        results = evaluator.comprehensive_evaluation(model_path, "custom_cnn")
    else:
        print("Model file not found. Train a model first using train.py")
        
        # Demonstrate individual evaluation functions
        print("\nDemonstrating evaluation functions...")
        evaluator = ModelEvaluator()
        
        # Show dataset statistics
        stats = evaluator.preprocessor.get_dataset_statistics()
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
