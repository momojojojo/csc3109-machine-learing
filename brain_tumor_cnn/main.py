"""
Main execution script for Brain Tumor CNN Classification
Provides a comprehensive interface to run the entire pipeline
"""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf

# Import project modules
from config import config
from data_preprocessing import DataPreprocessor
from model import BrainTumorDenseNet
from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import VisualizationUtils, DataAnalysisUtils, ModelUtils

class BrainTumorClassificationPipeline:
    def __init__(self):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.viz_utils = VisualizationUtils()
        self.data_utils = DataAnalysisUtils()
        self.model_utils = ModelUtils()
        
        # Create directories
        self.config.create_directories()
        
    def run_data_analysis(self):
        """Run comprehensive data analysis"""
        print("Running Data Analysis...")
        print("=" * 50)
        
        # Get dataset statistics
        stats = self.preprocessor.get_dataset_statistics()
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Analyze dataset balance
        df = self.preprocessor.load_dataset_info()
        balance_analysis = self.data_utils.analyze_dataset_balance(df)
        
        # Visualize data
        print("\nGenerating visualizations...")
        self.preprocessor.visualize_class_distribution()
        self.preprocessor.visualize_sample_images()
        
        return stats, balance_analysis
    
    def train_model(self, model_type='densenet', densenet_variant='DenseNet121', epochs=None):
        """Train a DenseNet model"""
        print(f"Training {model_type} model...")
        print("=" * 50)
        
        # Initialize trainer
        trainer = ModelTrainer(model_type=model_type, densenet_variant=densenet_variant)
        
        # Train model
        history, model_name = trainer.train_model(epochs=epochs)
        
        # Plot training history
        trainer.plot_training_history(model_name)
        
        # Get training summary
        summary = trainer.get_training_summary()
        print("\nTraining Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
        
        return history, model_name, trainer.model
    
    def evaluate_model(self, model_path, model_name=None):
        """Evaluate a trained model"""
        print("Evaluating model...")
        print("=" * 50)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation(model_path, model_name)
        
        return results, evaluator
    
    def compare_models(self, model_paths, model_names=None):
        """Compare multiple trained models"""
        print("Comparing models...")
        print("=" * 50)
        
        if model_names is None:
            model_names = [os.path.basename(path).replace('.h5', '') for path in model_paths]
        
        results_dict = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            if os.path.exists(model_path):
                print(f"\nEvaluating {model_name}...")
                evaluator = ModelEvaluator()
                results = evaluator.comprehensive_evaluation(model_path, model_name)
                results_dict[model_name] = results['metrics']
            else:
                print(f"Model not found: {model_path}")
        
        # Create comparison visualizations
        if len(results_dict) > 1:
            self.viz_utils.plot_model_comparison(results_dict)
        
        return results_dict
    
    def run_full_pipeline(self, model_types=['custom'], epochs=None):
        """Run the complete pipeline from data analysis to model evaluation"""
        print("Running Full Brain Tumor Classification Pipeline")
        print("=" * 60)
        
        # Step 1: Data Analysis
        print("\n1. DATA ANALYSIS")
        stats, balance_analysis = self.run_data_analysis()
        
        # Step 2: Train Models
        print("\n2. MODEL TRAINING")
        trained_models = {}
        
        for model_type in model_types:
            print(f"\nTraining {model_type} model...")
            history, model_name, model = self.train_model(
                model_type=model_type, 
                epochs=epochs
            )
            
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{model_name}_best.h5")
            trained_models[model_name] = {
                'path': model_path,
                'history': history.history if history else None,
                'model': model
            }
        
        # Step 3: Model Evaluation
        print("\n3. MODEL EVALUATION")
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            if os.path.exists(model_info['path']):
                print(f"\nEvaluating {model_name}...")
                results, evaluator = self.evaluate_model(model_info['path'], model_name)
                evaluation_results[model_name] = results
        
        # Step 4: Model Comparison
        if len(trained_models) > 1:
            print("\n4. MODEL COMPARISON")
            model_paths = [info['path'] for info in trained_models.values()]
            model_names = list(trained_models.keys())
            comparison_results = self.compare_models(model_paths, model_names)
        
        # Step 5: Generate Final Report
        print("\n5. GENERATING FINAL REPORT")
        self.generate_final_report(stats, trained_models, evaluation_results)
        
        print("\nPipeline completed successfully!")
        return trained_models, evaluation_results
    
    def generate_final_report(self, stats, trained_models, evaluation_results):
        """Generate a comprehensive final report"""
        report_path = os.path.join(self.config.RESULTS_PATH, 'final_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("BRAIN TUMOR CNN CLASSIFICATION - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset Information
            f.write("1. DATASET INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images: {stats['total_images']}\n")
            f.write(f"Classes: {', '.join(stats['classes'])}\n")
            f.write("Class Distribution:\n")
            for class_name, count in stats['class_distribution'].items():
                f.write(f"  {class_name}: {count}\n")
            f.write("\n")
            
            # Model Performance
            f.write("2. MODEL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            for model_name, results in evaluation_results.items():
                f.write(f"\n{model_name.upper()}:\n")
                if 'metrics' in results and 'overall' in results['metrics']:
                    metrics = results['metrics']['overall']
                    f.write(f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"  Precision: {metrics.get('precision_macro', 0):.4f}\n")
                    f.write(f"  Recall: {metrics.get('recall_macro', 0):.4f}\n")
                    f.write(f"  F1-Score: {metrics.get('f1_macro', 0):.4f}\n")
            
            # Model Architecture
            f.write("\n3. MODEL ARCHITECTURES\n")
            f.write("-" * 30 + "\n")
            
            for model_name, model_info in trained_models.items():
                if model_info['model'] is not None:
                    f.write(f"\n{model_name.upper()}:\n")
                    total_params = model_info['model'].count_params()
                    f.write(f"  Total Parameters: {total_params:,}\n")
            
            # Recommendations
            f.write("\n4. RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("Based on the evaluation results:\n")
            
            if evaluation_results:
                best_model = max(evaluation_results.items(), 
                               key=lambda x: x[1]['metrics']['overall'].get('accuracy', 0))
                f.write(f"- Best performing model: {best_model[0]}\n")
                f.write(f"- Best accuracy achieved: {best_model[1]['metrics']['overall'].get('accuracy', 0):.4f}\n")
            
            f.write("- Consider data augmentation for better generalization\n")
            f.write("- Monitor for overfitting during training\n")
            f.write("- Validate results with medical professionals\n")
            
            f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Final report saved to: {report_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Brain Tumor DenseNet Classification Pipeline')
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['analysis', 'train', 'evaluate', 'compare', 'full'],
                       help='Pipeline mode to run')
    
    parser.add_argument('--model-type', type=str, default='densenet',
                       choices=['densenet', 'densenet_transfer', 'densenet_light'],
                       help='Type of DenseNet model to train')
    
    parser.add_argument('--densenet-variant', type=str, default='DenseNet121',
                       choices=['DenseNet121', 'DenseNet169', 'DenseNet201'],
                       help='DenseNet variant for transfer learning')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model for evaluation')
    
    parser.add_argument('--model-paths', type=str, nargs='+', default=None,
                       help='Paths to multiple models for comparison')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BrainTumorClassificationPipeline()
    
    # Run based on mode
    if args.mode == 'analysis':
        pipeline.run_data_analysis()
        
    elif args.mode == 'train':
        pipeline.train_model(
            model_type=args.model_type,
            densenet_variant=args.densenet_variant,
            epochs=args.epochs
        )
        
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
            sys.exit(1)
        pipeline.evaluate_model(args.model_path)
        
    elif args.mode == 'compare':
        if args.model_paths is None:
            print("Error: --model-paths required for comparison mode")
            sys.exit(1)
        pipeline.compare_models(args.model_paths)
        
    elif args.mode == 'full':
        model_types = [args.model_type]
        pipeline.run_full_pipeline(model_types=model_types, epochs=args.epochs)

if __name__ == "__main__":
    # Set up TensorFlow
    print("Brain Tumor CNN Classification System")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")
    
    print()
    
    # Run main function
    main()
