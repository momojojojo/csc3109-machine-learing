"""
Data preprocessing module for Brain Tumor CNN Classification
Handles data loading, preprocessing, augmentation, and dataset creation
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from config import config

class DataPreprocessor:
    def __init__(self):
        self.config = config
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        
    def load_dataset_info(self):
        """Load dataset information and create dataframe with image paths and labels"""
        data = []
        
        for class_idx, class_name in enumerate(self.config.CLASS_NAMES):
            class_path = os.path.join(self.config.DATASET_PATH, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} not found!")
                continue
                
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    data.append({
                        'image_path': image_path,
                        'class_name': class_name,
                        'class_label': class_idx
                    })
        
        df = pd.DataFrame(data)
        print(f"Dataset loaded: {len(df)} images")
        print(f"Class distribution:")
        print(df['class_name'].value_counts())
        
        return df
    
    def preprocess_image(self, image_path, target_size=None):
        """Preprocess a single image"""
        if target_size is None:
            target_size = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def create_data_generators(self):
        """Create data generators for training, validation, and testing"""
        
        # Load dataset information
        df = self.load_dataset_info()
        
        # Split dataset into train, validation, and test sets
        train_df, temp_df = train_test_split(
            df, 
            test_size=(self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT),
            stratify=df['class_label'],
            random_state=42
        )
        
        val_size = self.config.VALIDATION_SPLIT / (self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['class_label'],
            random_state=42
        )
        
        print(f"Train set: {len(train_df)} images")
        print(f"Validation set: {len(val_df)} images")
        print(f"Test set: {len(test_df)} images")
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            fill_mode=self.config.FILL_MODE
        )
        
        # Validation and test generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators from dataframes
        self.train_generator = self.create_generator_from_df(
            train_df, train_datagen, shuffle=True
        )
        
        self.validation_generator = self.create_generator_from_df(
            val_df, val_test_datagen, shuffle=False
        )
        
        self.test_generator = self.create_generator_from_df(
            test_df, val_test_datagen, shuffle=False
        )
        
        return self.train_generator, self.validation_generator, self.test_generator
    
    def create_generator_from_df(self, df, datagen, shuffle=True):
        """Create a data generator from a dataframe"""
        
        def data_generator():
            indices = np.arange(len(df))
            if shuffle:
                np.random.shuffle(indices)
                
            for start_idx in range(0, len(df), self.config.BATCH_SIZE):
                end_idx = min(start_idx + self.config.BATCH_SIZE, len(df))
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    row = df.iloc[idx]
                    image = self.preprocess_image(row['image_path'])
                    
                    if image is not None:
                        batch_images.append(image)
                        batch_labels.append(row['class_label'])
                
                if batch_images:
                    batch_images = np.array(batch_images)
                    batch_labels = to_categorical(batch_labels, num_classes=self.config.NUM_CLASSES)
                    
                    yield batch_images, batch_labels
        
        return data_generator
    
    def get_dataset_statistics(self):
        """Get dataset statistics"""
        df = self.load_dataset_info()
        
        stats = {
            'total_images': len(df),
            'classes': self.config.CLASS_NAMES,
            'class_distribution': df['class_name'].value_counts().to_dict(),
            'images_per_class': df.groupby('class_name').size().to_dict()
        }
        
        return stats
    
    def visualize_sample_images(self, num_samples=8):
        """Visualize sample images from each class"""
        df = self.load_dataset_info()
        
        fig, axes = plt.subplots(len(self.config.CLASS_NAMES), num_samples//len(self.config.CLASS_NAMES), 
                                figsize=(15, 12))
        fig.suptitle('Sample Images from Each Class', fontsize=16)
        
        for class_idx, class_name in enumerate(self.config.CLASS_NAMES):
            class_df = df[df['class_name'] == class_name].sample(n=num_samples//len(self.config.CLASS_NAMES))
            
            for img_idx, (_, row) in enumerate(class_df.iterrows()):
                image = self.preprocess_image(row['image_path'])
                
                if image is not None:
                    axes[class_idx, img_idx].imshow(image)
                    axes[class_idx, img_idx].set_title(f'{class_name}')
                    axes[class_idx, img_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'sample_images.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_class_distribution(self):
        """Visualize class distribution"""
        stats = self.get_dataset_statistics()
        
        plt.figure(figsize=(10, 6))
        classes = list(stats['class_distribution'].keys())
        counts = list(stats['class_distribution'].values())
        
        plt.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Brain Tumor Classes')
        plt.ylabel('Number of Images')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 1, str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'class_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create directories
    config.create_directories()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get dataset statistics
    stats = preprocessor.get_dataset_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize data
    preprocessor.visualize_class_distribution()
    preprocessor.visualize_sample_images()
    
    # Create data generators
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    print("Data generators created successfully!")
