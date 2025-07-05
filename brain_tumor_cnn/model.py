"""
DenseNet CNN Model Architecture for Brain Tumor Classification
Implements DenseNet architecture - different from VGG19 and ResNet used by teammates
DenseNet uses dense connections where each layer connects to every other layer in a feed-forward fashion
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, 
    BatchNormalization, GlobalAveragePooling2D, Input,
    AveragePooling2D, Concatenate, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from config import config

class BrainTumorDenseNet:
    """
    DenseNet-based CNN for Brain Tumor Classification
    
    DenseNet is DIFFERENT from VGG19 and ResNet:
    - VGG19: Uses sequential layers with simple convolutions
    - ResNet: Uses skip connections (residual connections)
    - DenseNet: Uses DENSE connections where each layer connects to ALL previous layers
    
    This creates feature reuse and reduces vanishing gradient problem differently than ResNet
    """
    def __init__(self):
        self.config = config
        self.model = None
    
    def dense_block(self, x, num_layers, growth_rate, name):
        """
        Create a dense block - the core of DenseNet architecture
        Each layer in the block receives feature maps from ALL preceding layers
        """
        for i in range(num_layers):
            # Bottleneck layer (1x1 conv to reduce channels)
            bn1 = BatchNormalization(name=f'{name}_bn1_{i}')(x)
            act1 = Activation('relu', name=f'{name}_relu1_{i}')(bn1)
            conv1 = Conv2D(4 * growth_rate, (1, 1), 
                          kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                          name=f'{name}_conv1_{i}')(act1)
            
            # 3x3 convolution
            bn2 = BatchNormalization(name=f'{name}_bn2_{i}')(conv1)
            act2 = Activation('relu', name=f'{name}_relu2_{i}')(bn2)
            conv2 = Conv2D(growth_rate, (3, 3), padding='same',
                          kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                          name=f'{name}_conv2_{i}')(act2)
            
            # Dense connection: concatenate with input
            x = Concatenate(name=f'{name}_concat_{i}')([x, conv2])
        
        return x
    
    def transition_block(self, x, compression_factor, name):
        """
        Transition block between dense blocks
        Reduces feature map size and number of channels
        """
        # Get number of input channels
        num_channels = int(x.shape[-1] * compression_factor)
        
        # Batch norm and activation
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = Activation('relu', name=f'{name}_relu')(x)
        
        # 1x1 convolution to reduce channels
        x = Conv2D(num_channels, (1, 1),
                  kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                  name=f'{name}_conv')(x)
        
        # Average pooling to reduce spatial dimensions
        x = AveragePooling2D((2, 2), strides=2, name=f'{name}_pool')(x)
        
        return x
    
    def create_densenet_model(self, growth_rate=32, num_blocks=[6, 12, 24, 16], compression_factor=0.5):
        """
        Create custom DenseNet architecture for brain tumor classification
        
        Args:
            growth_rate: Number of feature maps added by each layer in dense block
            num_blocks: Number of layers in each dense block
            compression_factor: Factor to reduce channels in transition blocks
        """
        inputs = Input(shape=self.config.INPUT_SHAPE)
        
        # Initial convolution
        x = Conv2D(64, (7, 7), strides=2, padding='same',
                  kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                  name='initial_conv')(inputs)
        x = BatchNormalization(name='initial_bn')(x)
        x = Activation('relu', name='initial_relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same', name='initial_pool')(x)
        
        # Dense blocks with transition blocks
        for i, num_layers in enumerate(num_blocks):
            # Dense block
            x = self.dense_block(x, num_layers, growth_rate, f'dense_block_{i+1}')
            
            # Transition block (except after last dense block)
            if i < len(num_blocks) - 1:
                x = self.transition_block(x, compression_factor, f'transition_{i+1}')
        
        # Final batch norm and activation
        x = BatchNormalization(name='final_bn')(x)
        x = Activation('relu', name='final_relu')(x)
        
        # Global average pooling
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dropout for regularization
        x = Dropout(self.config.DROPOUT_RATE, name='dropout')(x)
        
        # Classification head
        x = Dense(512, activation='relu',
                 kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                 name='fc1')(x)
        x = BatchNormalization(name='fc1_bn')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='fc1_dropout')(x)
        
        x = Dense(256, activation='relu',
                 kernel_regularizer=l2(self.config.L2_REGULARIZATION),
                 name='fc2')(x)
        x = BatchNormalization(name='fc2_bn')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='fc2_dropout')(x)
        
        # Output layer
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        model = Model(inputs, outputs, name='BrainTumor_DenseNet')
        return model
    
    def create_densenet_transfer_learning(self, densenet_variant='DenseNet121'):
        """
        Create DenseNet model using transfer learning from ImageNet
        
        Args:
            densenet_variant: 'DenseNet121', 'DenseNet169', or 'DenseNet201'
        """
        # Choose DenseNet variant
        if densenet_variant == 'DenseNet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.INPUT_SHAPE
            )
        elif densenet_variant == 'DenseNet169':
            base_model = DenseNet169(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.INPUT_SHAPE
            )
        elif densenet_variant == 'DenseNet201':
            base_model = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.INPUT_SHAPE
            )
        else:
            raise ValueError(f"Unsupported DenseNet variant: {densenet_variant}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = Input(shape=self.config.INPUT_SHAPE)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense layers for classification
        x = Dense(1024, activation='relu',
                 kernel_regularizer=l2(self.config.L2_REGULARIZATION))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        
        x = Dense(512, activation='relu',
                 kernel_regularizer=l2(self.config.L2_REGULARIZATION))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'BrainTumor_{densenet_variant}_Transfer')
        return model
    
    def create_lightweight_densenet(self):
        """
        Create a lightweight version of DenseNet for faster training
        """
        inputs = Input(shape=self.config.INPUT_SHAPE)
        
        # Initial convolution (smaller)
        x = Conv2D(32, (3, 3), padding='same',
                  kernel_regularizer=l2(self.config.L2_REGULARIZATION))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Smaller dense blocks
        growth_rate = 16
        num_blocks = [4, 6, 8]
        
        for i, num_layers in enumerate(num_blocks):
            # Dense block
            x = self.dense_block(x, num_layers, growth_rate, f'lite_dense_block_{i+1}')
            
            # Transition block (except after last dense block)
            if i < len(num_blocks) - 1:
                x = self.transition_block(x, 0.5, f'lite_transition_{i+1}')
        
        # Final processing
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        
        # Simple classification head
        x = Dense(256, activation='relu',
                 kernel_regularizer=l2(self.config.L2_REGULARIZATION))(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='BrainTumor_LightDenseNet')
        return model
    
    def build_model(self, model_type='densenet', densenet_variant='DenseNet121'):
        """
        Build and compile the DenseNet model
        
        Args:
            model_type: 'densenet', 'densenet_transfer', or 'densenet_light'
            densenet_variant: For transfer learning - 'DenseNet121', 'DenseNet169', 'DenseNet201'
        """
        
        if model_type == 'densenet':
            self.model = self.create_densenet_model()
            print("Created custom DenseNet model")
        elif model_type == 'densenet_transfer':
            self.model = self.create_densenet_transfer_learning(densenet_variant)
            print(f"Created DenseNet transfer learning model with {densenet_variant}")
        elif model_type == 'densenet_light':
            self.model = self.create_lightweight_densenet()
            print("Created lightweight DenseNet model")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary and architecture details"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        print("Model Summary:")
        print("=" * 50)
        self.model.summary()
        
        # Calculate total parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nModel Parameters:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params
        }
    
    def save_model_architecture(self, filepath=None):
        """Save model architecture as image"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        if filepath is None:
            filepath = f"{self.config.RESULTS_PATH}/model_architecture.png"
        
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96
            )
            print(f"Model architecture saved to: {filepath}")
        except Exception as e:
            print(f"Could not save model architecture: {str(e)}")
    
    def fine_tune_model(self, base_layers_to_unfreeze=10):
        """Fine-tune a transfer learning model by unfreezing top layers"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        # Find the base model layer
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break
        
        if base_model is None:
            print("No base model found for fine-tuning")
            return
        
        # Unfreeze top layers
        base_model.trainable = True
        for layer in base_model.layers[:-base_layers_to_unfreeze]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"Fine-tuning enabled: unfroze top {base_layers_to_unfreeze} layers")
        print(f"New learning rate: {self.config.LEARNING_RATE / 10}")

# Example usage
if __name__ == "__main__":
    # Create model instance
    cnn_model = BrainTumorCNN()
    
    # Build different types of models
    print("Building Custom CNN...")
    model1 = cnn_model.build_model(model_type='custom')
    cnn_model.get_model_summary()
    
    print("\n" + "="*60 + "\n")
    
    print("Building Transfer Learning Model...")
    cnn_model2 = BrainTumorCNN()
    model2 = cnn_model2.build_model(model_type='transfer', base_model_name='VGG16')
    cnn_model2.get_model_summary()
    
    print("\n" + "="*60 + "\n")
    
    print("Building Lightweight CNN...")
    cnn_model3 = BrainTumorCNN()
    model3 = cnn_model3.build_model(model_type='lightweight')
    cnn_model3.get_model_summary()
