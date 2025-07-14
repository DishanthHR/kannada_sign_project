import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_lite_transformer_model(input_shape, num_classes):
    """
    Create a lightweight transformer model for sign language recognition.
    
    Args:
        input_shape: Shape of input data (frames, hands, landmarks, coordinates)
        num_classes: Number of output classes
        
    Returns:
        model: Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape to (batch, frames, features)
    # Combine hands and landmarks into a single feature dimension
    x = layers.Reshape((input_shape[0], -1))(inputs)
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], x.shape[-1])
    x = x + pos_encoding
    
    # Transformer blocks
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformer encoder block.
    
    Args:
        inputs: Input tensor
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network
        dropout: Dropout rate
        
    Returns:
        x: Output tensor
    """
    # Multi-head attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # Feed forward network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    
    return x + res

def positional_encoding(length, depth):
    """
    Generate positional encoding for transformer.
    
    Args:
        length: Sequence length
        depth: Embedding depth
        
    Returns:
        pos_encoding: Positional encoding tensor
    """
    # Generate position indices
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    
    # Generate dimension indices
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
    
    # Calculate angle rates
    angle_rates = 1 / (10000**depths)
    
    # Calculate angles
    angle_rads = positions * angle_rates
    
    # Apply sin to even indices and cos to odd indices
    pos_encoding = tf.concat(
        [tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])],
        axis=-1
    )
    
    # Add batch dimension
    pos_encoding = pos_encoding[tf.newaxis, ...]
    
    return tf.cast(pos_encoding, tf.float32)

def create_cnn_lstm_model(input_shape, num_classes):
    """
    Create a CNN-LSTM model for sign language recognition.
    
    Args:
        input_shape: Shape of input data (frames, hands, landmarks, coordinates)
        num_classes: Number of output classes
        
    Returns:
        model: Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape to (batch, frames, hands*landmarks*coordinates)
    x = layers.Reshape((input_shape[0], -1))(inputs)
    
    # 1D CNN layers
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
