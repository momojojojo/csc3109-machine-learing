"""
Configuration file for Brain Tumor CNN Classification
Contains all hyperparameters and settings for the project
"""

import os

class Config:
    # Dataset paths
    DATASET_PATH = "../dataset_35/dataset_8"
    MODEL_SAVE_PATH = "models/"
    RESULTS_PATH = "results/"
    LOGS_PATH = "logs/"
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    # Dataset parameters
    NUM_CLASSES = 4
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Data augmentation parameters
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True
    FILL_MODE = 'nearest'
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.001
    
    # Training callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.2
    MIN_LR = 1e-7
    
    # Evaluation parameters
    CONFIDENCE_THRESHOLD = 0.5
    
    # Create directories
    @staticmethod
    def create_directories():
        """Create necessary directories for the project"""
        directories = [
            Config.MODEL_SAVE_PATH,
            Config.RESULTS_PATH,
            Config.LOGS_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

# Global configuration instance
config = Config()
