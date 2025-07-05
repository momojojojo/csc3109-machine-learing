# Brain Tumor DenseNet Classification

A comprehensive deep learning solution for classifying brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor using **DenseNet** architecture with TensorFlow/Keras.

## Project Overview

This project implements a complete machine learning pipeline for brain tumor classification using **DenseNet**, which is distinctly different from VGG19 and ResNet architectures:

**Why DenseNet?**
- **Dense Connections**: Each layer connects to ALL previous layers, creating maximum information flow
- **Feature Reuse**: Reduces redundancy and improves gradient flow
- **Parameter Efficiency**: Fewer parameters than traditional CNNs while maintaining performance
- **Different from teammates' approaches**: VGG19 (sequential), ResNet (skip connections), DenseNet (dense connections)

**Features:**
- Custom DenseNet implementation for medical imaging
- DenseNet transfer learning (DenseNet121, DenseNet169, DenseNet201)
- Lightweight DenseNet for faster training
- Comprehensive model training and validation
- Detailed evaluation with multiple metrics
- Advanced visualization and analysis tools

## Dataset

The project works with brain MRI images organized in the following structure:
```
dataset_35/dataset_8/
├── glioma/          # Glioma tumor images
├── meningioma/      # Meningioma tumor images
├── notumor/         # No tumor (healthy) images
└── pituitary/       # Pituitary tumor images
```

## Project Structure

```
brain_tumor_cnn/
├── config.py                 # Configuration and hyperparameters
├── data_preprocessing.py     # Data loading and preprocessing
├── model.py                 # CNN architecture definitions
├── train.py                 # Training pipeline
├── evaluate.py              # Model evaluation and metrics
├── utils.py                 # Utility functions and visualizations
├── main.py                  # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # This file

Generated directories:
├── models/                  # Saved trained models
├── results/                 # Evaluation results and plots
└── logs/                   # Training logs and TensorBoard data
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify TensorFlow installation:**
```python
import tensorflow as tf
print(tf.__version__)
```

## Usage

### Quick Start

Run the complete pipeline with default settings:
```bash
cd brain_tumor_cnn
python main.py --mode full
```

### Command Line Options

The main script supports various modes and DenseNet options:

```bash
# Data analysis only
python main.py --mode analysis

# Train a custom DenseNet model
python main.py --mode train --model-type densenet --epochs 20

# Train with DenseNet transfer learning
python main.py --mode train --model-type densenet_transfer --densenet-variant DenseNet121 --epochs 15

# Train lightweight DenseNet
python main.py --mode train --model-type densenet_light --epochs 15

# Evaluate a trained model
python main.py --mode evaluate --model-path models/densenet_20240704_123456_best.h5

# Compare multiple models
python main.py --mode compare --model-paths models/model1.h5 models/model2.h5

# Full pipeline with specific settings
python main.py --mode full --model-type densenet --epochs 25
```

### Available DenseNet Model Types

1. **Custom DenseNet** (`--model-type densenet`)
   - Custom-designed DenseNet architecture
   - Dense connections between all layers
   - Optimized for medical image classification
   - Good balance of performance and efficiency

2. **DenseNet Transfer Learning** (`--model-type densenet_transfer`)
   - Uses pre-trained DenseNet models from ImageNet
   - Available variants: DenseNet121, DenseNet169, DenseNet201
   - Faster training, often better performance
   - Example: `--densenet-variant DenseNet169`

3. **Lightweight DenseNet** (`--model-type densenet_light`)
   - Smaller, faster DenseNet model
   - Suitable for resource-constrained environments
   - Quick training and inference
   - Maintains dense connection benefits

### Python API Usage

You can also use the components programmatically:

```python
from brain_tumor_cnn.main import BrainTumorClassificationPipeline

# Initialize pipeline
pipeline = BrainTumorClassificationPipeline()

# Run data analysis
stats, balance = pipeline.run_data_analysis()

# Train a model
history, model_name, model = pipeline.train_model(
    model_type='custom',
    epochs=20
)

# Evaluate model
results, evaluator = pipeline.evaluate_model('models/model.h5')
```

## Configuration

Key parameters can be modified in `config.py`:

```python
# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data splits
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation
ROTATION_RANGE = 20
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
```

## Features

### Data Preprocessing
- Automatic image loading and preprocessing
- Data augmentation for better generalization
- Train/validation/test split with stratification
- Class distribution analysis

### Model Architectures
- **Custom CNN**: Multi-layer architecture with batch normalization and dropout
- **Transfer Learning**: Pre-trained models with custom classification heads
- **Lightweight**: Efficient architecture for fast inference

### Training Features
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- TensorBoard logging
- Comprehensive training history tracking

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class and overall metrics
- Confusion matrix visualization
- ROC curves and AUC scores
- Sample prediction visualization

### Advanced Visualizations
- Class activation maps (CAM)
- Feature distribution plots
- Training history comparisons
- Interactive confusion matrices
- Model performance comparisons

## Output Files

The pipeline generates various output files:

### Models
- `models/[model_name]_best.h5` - Best model during training
- `models/[model_name]_final.h5` - Final model after training

### Results
- `results/[model_name]_confusion_matrix.png` - Confusion matrix
- `results/[model_name]_roc_curves.png` - ROC curves
- `results/[model_name]_training_history.png` - Training plots
- `results/[model_name]_evaluation_results.json` - Detailed metrics
- `results/class_distribution.png` - Dataset distribution
- `results/sample_images.png` - Sample images from each class

### Logs
- `logs/[model_name]/` - TensorBoard logs
- `logs/[model_name]_training.csv` - Training metrics CSV

### Reports
- `results/final_report.txt` - Comprehensive final report

## Performance Expectations

Typical performance on brain tumor classification:

| Model Type | Accuracy | Training Time | Parameters |
|------------|----------|---------------|------------|
| Custom CNN | 85-92% | 30-60 min | ~2M |
| Transfer Learning (VGG16) | 90-95% | 20-40 min | ~15M |
| Lightweight | 80-88% | 15-30 min | ~500K |

*Performance may vary based on dataset size, hardware, and hyperparameters.*

## Hardware Requirements

### Minimum Requirements
- RAM: 8GB
- Storage: 5GB free space
- CPU: Multi-core processor

### Recommended Requirements
- RAM: 16GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 10GB+ free space
- CUDA-compatible GPU for faster training

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in `config.py`
   - Use lightweight model
   - Close other applications

2. **Slow Training**
   - Enable GPU acceleration
   - Reduce image size
   - Use transfer learning

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Try different model architectures
   - Check data quality and balance

4. **Import Errors**
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Ensure TensorFlow is properly installed

### GPU Setup

For NVIDIA GPUs:
```bash
# Install CUDA-compatible TensorFlow
pip install tensorflow-gpu

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Contributing

To extend or modify the project:

1. **Adding New Models**: Implement in `model.py`
2. **Custom Metrics**: Add to `evaluate.py`
3. **New Visualizations**: Extend `utils.py`
4. **Data Augmentation**: Modify `data_preprocessing.py`

## Medical Disclaimer

This tool is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and approval from medical professionals. Always consult qualified healthcare providers for medical decisions.

## License

This project is for educational purposes. Please ensure compliance with your institution's policies and any applicable licenses for the dataset used.

## Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Medical imaging community for dataset contributions
- Open source contributors for various utilities and tools

## Contact

For questions or issues, please refer to the project documentation or create an issue in the project repository.

---

**Note**: This implementation is part of a machine learning course project and demonstrates best practices in deep learning for medical image classification.
