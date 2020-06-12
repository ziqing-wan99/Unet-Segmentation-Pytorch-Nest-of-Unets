import numpy as np
from scipy import spatial


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool) # prediction 
    im2 = np.asarray(im2).astype(np.bool) # true

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5 
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum() # prediction+ground truth label 
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2) # prediction=ground truth label, correct prediction 
    #print(im_sum)

    return 2. * intersection.sum() / im_sum # 2*correct prediction/(prediction+ground truth label)


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN
    Confusion matrix"""
    

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N) # accuracy 
    return accuracy * 100.0
