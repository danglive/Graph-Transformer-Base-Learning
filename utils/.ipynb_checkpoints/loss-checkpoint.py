# utils/loss.py

import torch.nn.functional as F


def multi_classification_loss(outputs, targets):
    """
    Computes the cross-entropy loss for multi-class classification.

    Args:
        outputs: The model predictions.
        targets: The ground truth labels.

    Returns:
        loss: The computed loss value.
    """
    return F.cross_entropy(outputs, targets)


def mse_loss(outputs, targets):
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        outputs: The model predictions.
        targets: The ground truth values.

    Returns:
        loss: The computed loss value.
    """
    return F.mse_loss(outputs, targets)


def multi_label_classification_loss(outputs, targets):
    """
    Computes the binary cross-entropy loss for multi-label classification.

    Args:
        outputs: The model predictions.
        targets: The ground truth labels.

    Returns:
        loss: The computed loss value.
    """
    return F.binary_cross_entropy_with_logits(outputs, targets)