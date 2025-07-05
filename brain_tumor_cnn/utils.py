"""
Utility functions for Brain Tumor CNN Classification
Contains helper functions for visualization, data analysis, and model utilities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import config

class VisualizationUtils:
    def __init__(self):
        self.config = config
        
    def plot_model_comparison(self, results_dict, save_plot=True):
        """Compare multiple models' performance"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Extract metrics for each model
        data = []
        for model_name, results in results_dict.items():
            for metric in metrics:
                if 'overall' in results and metric in results['overall']:
                    data.append({
                        'Model': model_name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': results['overall'][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.config.RESULTS_PATH, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_training_comparison(self, history_dict, save_plot=True):
        """Compare training histories of multiple models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'loss', 'precision', 'recall']
        titles = ['Accuracy', 'Loss', 'Precision', 'Recall']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for model_name, history in history_dict.items():
                if metric in history:
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(epochs, history[metric], label=f'{model_name} (train)', alpha=0.7)
                    
                    val_metric = f'val_{metric}'
                    if val_metric in history:
                        ax.plot(epochs, history[val_metric], label=f'{model_name} (val)', 
                               linestyle='--', alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.config.RESULTS_PATH, 'training_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_class_activation_maps(self, model, image_path, layer_name=None, save_plot=True):
        """Generate and plot class activation maps (CAM)"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            img_array = np.expand_dims(img_resized / 255.0, axis=0)
            
            # Get the last convolutional layer if not specified
            if layer_name is None:
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        layer_name = layer.name
                        break
            
            if layer_name is None:
                print("No convolutional layer found for CAM generation")
                return
            
            # Create a model that outputs the feature maps
            cam_model = tf.keras.Model(
                inputs=model.input,
                outputs=[model.get_layer(layer_name).output, model.output]
            )
            
            # Get feature maps and predictions
            with tf.GradientTape() as tape:
                conv_outputs, predictions = cam_model(img_array)
                predicted_class = tf.argmax(predictions[0])
                class_output = predictions[:, predicted_class]
            
            # Calculate gradients
            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap_resized, cmap='jet')
            axes[1].set_title('Class Activation Map')
            axes[1].axis('off')
            
            # Overlay
            overlay = img * 0.6 + np.expand_dims(heatmap_resized, axis=-1) * 0.4 * 255
            overlay = overlay.astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            # Add prediction info
            pred_class = self.config.CLASS_NAMES[predicted_class.numpy()]
            confidence = tf.nn.softmax(predictions[0])[predicted_class].numpy()
            fig.suptitle(f'Predicted: {pred_class} (Confidence: {confidence:.3f})', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plot:
                filename = os.path.basename(image_path).replace('.jpg', '_cam.png')
                plot_path = os.path.join(self.config.RESULTS_PATH, filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"CAM visualization saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating CAM: {str(e)}")
    
    def plot_feature_distribution(self, model, data_generator, layer_name=None, save_plot=True):
        """Plot feature distribution from a specific layer"""
        try:
            # Get a batch of data
            batch_images, batch_labels = next(data_generator())
            
            # Get the layer for feature extraction
            if layer_name is None:
                # Use the layer before the final classification layer
                layer_name = model.layers[-3].name
            
            # Create feature extraction model
            feature_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
            
            # Extract features
            features = feature_model.predict(batch_images, verbose=0)
            
            # Flatten features if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Get true labels
            true_labels = np.argmax(batch_labels, axis=1)
            
            # Apply dimensionality reduction
            if features.shape[1] > 50:
                pca = PCA(n_components=50)
                features_reduced = pca.fit_transform(features)
            else:
                features_reduced = features
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            features_2d = tsne.fit_transform(features_reduced)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            colors = ['red', 'blue', 'green', 'orange']
            for i, class_name in enumerate(self.config.CLASS_NAMES):
                mask = true_labels == i
                if np.any(mask):
                    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=colors[i], label=class_name, alpha=0.7, s=50)
            
            plt.title(f'Feature Distribution - {layer_name}', fontsize=16, fontweight='bold')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plot:
                plot_path = os.path.join(self.config.RESULTS_PATH, f'feature_distribution_{layer_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Feature distribution plot saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting feature distribution: {str(e)}")
    
    def create_interactive_confusion_matrix(self, y_true, y_pred, save_plot=True):
        """Create interactive confusion matrix using Plotly"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create hovertext
        hovertext = []
        for i in range(len(cm)):
            hovertext.append([])
            for j in range(len(cm[0])):
                hovertext[-1].append(f'Count: {cm[i][j]}<br>Percentage: {cm_percent[i][j]:.1f}%')
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.config.CLASS_NAMES,
            y=self.config.CLASS_NAMES,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>',
            text=hovertext,
            texttemplate='%{z}',
            textfont={"size": 12},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Interactive Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=600,
            height=600
        )
        
        if save_plot:
            plot_path = os.path.join(self.config.RESULTS_PATH, 'interactive_confusion_matrix.html')
            fig.write_html(plot_path)
            print(f"Interactive confusion matrix saved to: {plot_path}")
        
        fig.show()
        
        return fig

class DataAnalysisUtils:
    def __init__(self):
        self.config = config
    
    def analyze_dataset_balance(self, data_df):
        """Analyze dataset class balance"""
        class_counts = data_df['class_name'].value_counts()
        total_samples = len(data_df)
        
        analysis = {
            'total_samples': total_samples,
            'class_counts': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            'balance_ratio': class_counts.max() / class_counts.min(),
            'is_balanced': class_counts.max() / class_counts.min() <= 2.0
        }
        
        print("Dataset Balance Analysis:")
        print("=" * 40)
        print(f"Total samples: {total_samples}")
        print(f"Balance ratio: {analysis['balance_ratio']:.2f}")
        print(f"Is balanced: {analysis['is_balanced']}")
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = count / total_samples * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return analysis
    
    def calculate_image_statistics(self, image_paths, sample_size=100):
        """Calculate statistics about image properties"""
        if len(image_paths) > sample_size:
            sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        else:
            sample_paths = image_paths
        
        widths, heights, channels = [], [], []
        mean_pixels, std_pixels = [], []
        
        for path in sample_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w, c = img.shape
                    heights.append(h)
                    widths.append(w)
                    channels.append(c)
                    
                    # Calculate pixel statistics
                    img_normalized = img.astype(np.float32) / 255.0
                    mean_pixels.append(np.mean(img_normalized))
                    std_pixels.append(np.std(img_normalized))
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
        
        stats = {
            'width': {'mean': np.mean(widths), 'std': np.std(widths), 'min': np.min(widths), 'max': np.max(widths)},
            'height': {'mean': np.mean(heights), 'std': np.std(heights), 'min': np.min(heights), 'max': np.max(heights)},
            'channels': {'mode': max(set(channels), key=channels.count)},
            'pixel_intensity': {'mean': np.mean(mean_pixels), 'std': np.mean(std_pixels)}
        }
        
        print("Image Statistics:")
        print("=" * 30)
        print(f"Width: {stats['width']['mean']:.1f} ± {stats['width']['std']:.1f} (range: {stats['width']['min']}-{stats['width']['max']})")
        print(f"Height: {stats['height']['mean']:.1f} ± {stats['height']['std']:.1f} (range: {stats['height']['min']}-{stats['height']['max']})")
        print(f"Channels: {stats['channels']['mode']}")
        print(f"Mean pixel intensity: {stats['pixel_intensity']['mean']:.3f} ± {stats['pixel_intensity']['std']:.3f}")
        
        return stats

class ModelUtils:
    def __init__(self):
        self.config = config
    
    def compare_model_sizes(self, model_paths):
        """Compare file sizes of different models"""
        model_info = {}
        
        for path in model_paths:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                model_name = os.path.basename(path).replace('.h5', '')
                
                # Load model to get parameter count
                try:
                    model = tf.keras.models.load_model(path)
                    param_count = model.count_params()
                    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                    
                    model_info[model_name] = {
                        'file_size_mb': size_mb,
                        'total_params': param_count,
                        'trainable_params': trainable_params,
                        'non_trainable_params': param_count - trainable_params
                    }
                except Exception as e:
                    print(f"Error loading model {path}: {str(e)}")
        
        # Create comparison DataFrame
        df = pd.DataFrame(model_info).T
        print("Model Size Comparison:")
        print("=" * 50)
        print(df.round(2))
        
        return df
    
    def export_model_summary(self, model, filepath=None):
        """Export detailed model summary to text file"""
        if filepath is None:
            filepath = os.path.join(self.config.RESULTS_PATH, 'model_summary.txt')
        
        with open(filepath, 'w') as f:
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("Model Architecture Summary")
            print("=" * 50)
            model.summary()
            
            print("\n\nModel Configuration:")
            print("=" * 30)
            print(f"Total parameters: {model.count_params():,}")
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Non-trainable parameters: {model.count_params() - trainable_params:,}")
            
            print("\n\nLayer Details:")
            print("=" * 30)
            for i, layer in enumerate(model.layers):
                print(f"Layer {i+1}: {layer.name}")
                print(f"  Type: {type(layer).__name__}")
                print(f"  Output shape: {layer.output_shape}")
                if hasattr(layer, 'activation'):
                    print(f"  Activation: {layer.activation.__name__}")
                print()
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"Model summary exported to: {filepath}")
        return filepath

# Example usage and utility functions
def load_and_preprocess_single_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_single_image(model, image_path, class_names):
    """Make prediction on a single image"""
    img = load_and_preprocess_single_image(image_path)
    
    if img is None:
        return None
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    result = {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': float(confidence),
        'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }
    
    return result

def create_prediction_report(results_list, output_path=None):
    """Create a comprehensive prediction report"""
    if output_path is None:
        output_path = os.path.join(config.RESULTS_PATH, 'prediction_report.txt')
    
    with open(output_path, 'w') as f:
        f.write("Brain Tumor Classification - Prediction Report\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results_list, 1):
            f.write(f"Prediction {i}:\n")
            f.write(f"  Image: {result.get('image_path', 'Unknown')}\n")
            f.write(f"  Predicted Class: {result['predicted_class']}\n")
            f.write(f"  Confidence: {result['confidence']:.4f}\n")
            f.write("  All Probabilities:\n")
            for class_name, prob in result['all_probabilities'].items():
                f.write(f"    {class_name}: {prob:.4f}\n")
            f.write("\n")
    
    print(f"Prediction report saved to: {output_path}")
    return output_path

# Example usage
if __name__ == "__main__":
    print("Brain Tumor CNN - Utility Functions")
    print("=" * 50)
    
    # Initialize utility classes
    viz_utils = VisualizationUtils()
    data_utils = DataAnalysisUtils()
    model_utils = ModelUtils()
    
    print("Utility classes initialized successfully!")
    print("\nAvailable utilities:")
    print("- VisualizationUtils: Advanced plotting and visualization")
    print("- DataAnalysisUtils: Dataset analysis and statistics")
    print("- ModelUtils: Model comparison and analysis")
    print("- Helper functions: Single image prediction, reporting")
