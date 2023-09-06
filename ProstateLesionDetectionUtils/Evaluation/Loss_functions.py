import tensorflow as tf

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """
    Compute the Dice coefficient between two tensors.
    
    Args:
    - y_true: Ground truth tensor.
    - y_pred: Predicted tensor.
    - epsilon: Small constant to avoid division by zero.
    
    Returns:
    - Dice coefficient.
    """
    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + epsilon) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)

def dice_loss(y_true, y_pred):
    """
    Compute the Dice loss between two tensors.
    
    Args:
    - y_true: Ground truth tensor.
    - y_pred: Predicted tensor.
    
    Returns:
    - Dice loss.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Compute the combined Dice and binary cross-entropy loss.
    
    Args:
    - y_true: Ground truth tensor.
    - y_pred: Predicted tensor.
    - alpha: Weight for the Dice loss.
    - beta: Weight for the binary cross-entropy loss.
    
    Returns:
    - Combined loss.
    """
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return alpha * dice + beta * bce