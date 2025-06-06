from sklearn.metrics import mean_squared_error
from math import log10, sqrt
import numpy as np
from prettytable import PrettyTable

from sewar.full_ref import rmse, ssim
from scipy import ndimage
EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
              - np.sum(s2 * np.log(s2)))
    return mi


# https://en.wikipedia.org/wiki/Confusion_matrix
def preValidation(actual, predict):
    if actual.shape != predict.shape:
        raise Exception("Actual and Predicted array shape must be equal")
    Max = 1
    if len(np.unique(actual)) == 2 and np.prod(np.unique(actual) == np.array([0, 1])):
        Max = np.unique(actual)[-1]
    elif not np.prod(np.unique(actual) == np.array([0, 1])):
        raise Exception("Actual Values are must be 0 and 1")
    if not np.prod(np.unique(predict) == np.array([0, 1])):
        raise Exception("Predicted Values are must be 0 and 1")
    return Max


def findConfusionMatrix(actual, predict, Max):
    act_one = np.where(actual == Max)
    act_zero = np.where(actual == 0)
    pred_one = np.where(predict == Max)
    pred_zero = np.where(predict == 0)

    '''Find Shape of the Each Dimension for Single Array Conversion'''
    array = [actual.shape[i] for i in range(len(actual.shape))]

    Act_One = np.zeros(shape=act_one[0].shape[0], dtype=np.int32)
    Act_Zero = np.zeros(shape=act_zero[0].shape[0], dtype=np.int32)
    Pred_One = np.zeros(shape=pred_one[0].shape[0], dtype=np.int32)
    Pred_Zero = np.zeros(shape=pred_zero[0].shape[0], dtype=np.int32)

    '''Convert Single Array for Easy Intersection'''
    for iter in range(len(act_one) - 1):
        Act_One += act_one[iter] * np.prod(array[iter + 1:])
        Act_Zero += act_zero[iter] * np.prod(array[iter + 1:])
        Pred_One += pred_one[iter] * np.prod(array[iter + 1:])
        Pred_Zero += pred_zero[iter] * np.prod(array[iter + 1:])
    Act_One += act_one[len(act_one) - 1]
    Act_Zero += act_zero[len(act_zero) - 1]
    Pred_One += pred_one[len(pred_one) - 1]
    Pred_Zero += pred_zero[len(pred_zero) - 1]

    '''Find Confusion Matrix'''
    # 1 ---> TP (True Positive) ------> If Actual = 1 and Predicted = 1
    TP = len(np.intersect1d(Act_One, Pred_One))
    # 2 ---> TN (True Negative) ------> If Actual = 0 and Predicted = 0
    TN = len(np.intersect1d(Act_Zero, Pred_Zero))
    # 3 ---> FP (False Positive) -----> If Actual = 0 and Predicted = 1
    FP = len(np.intersect1d(Act_Zero, Pred_One))
    # 4 ---> FN (False Negative) -----> If Actual = 1 and Predicted = 0
    FN = len(np.intersect1d(Act_One, Pred_Zero))

    # Positive P = TP + FN
    # Negative N = FP + TN
    # Predicted Positive PP = TP + FP
    # Predicted Negative PN = TN + FN
    # Total Population = P + N (or) PP + PN

    return array, [TP, TN, FP, FN]


def Accuracy(TP, TN, FP, FN):
    # Overall Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy * 100  # for Percentage


def Sensitivity(TP, FN):
    # Sensitivity, Hitrate, Recall, or True Positive Rate (TPR) = 1 - FNR
    # sensitivity = TP / P
    sensitivity = TP / (TP + FN)
    return sensitivity * 100  # for Percentage


def Specificity(TN, FP):
    # Specificity or True Negative Rate (TNR) = 1 - FPR
    # specificity = TN / N
    specificity = TN / (FP + TN)
    return specificity * 100  # for Percentage


def Precision(TP, FP):
    # Precision or Positive Predictive Value (PPV) = 1 - FDR
    # Precision = TP / PP
    precision = TP / (TP + FP)
    return precision * 100  # for Percentage


def FPR(TN, FP):
    # Fall out or False Positive Rate (FPR) = 1 - TNR
    # FPR = FP / N
    fpr = FP / (FP + TN)
    return fpr * 100  # for Percentage


def FNR(TP, FN):
    # False Negative Rate = 1 - TPR
    # FNR = FN / P
    fnr = FN / (TP + FN)
    return fnr * 100  # for Percentage


def NPV(TN, FN):
    # Negative Predictive Value (NPV) = 1- FOR
    # NPV = TN + PN
    npv = TN / (FN + TN)
    return npv * 100  # for Percentage


def FDR(TP, FP):
    # False Discovery Rate (FDR) = 1 - PPV
    # FDR = FP / PP
    fdr = FP / (TP + FP)
    return fdr * 100  # for Percentage


def F1SCORE(TP, FP, FN):
    # F1 score is the harmonic mean of Precision and Sensitivity
    # F1SCORE = (2 * PPV * TPR) / (PPV + TPR)
    f1score = (2 * TP) / (2 * TP + FP + FN)
    return f1score * 100  # for Percentage


def MCC(TP, TN, FP, FN):
    # Matthews Correlation Coefficient (MCC)
    # MCC = np.math.sqrt(TPR * TNR * PPV * NPV) - np.math.sqrt(FNR * FPR * FOR * FDR)
    mcc = ((TP * TN) - (FP * FN)) / np.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return mcc


def FOR(TN, FN):
    # False Omission Rate (FOR) = 1 - NPV
    # FOR = FN / PN
    For = FN / (FN + TN)
    return For * 100  # for Percentage


def PT(fpr, sensitivity):
    # Prevalence Threshold (PT)
    pt = (np.math.sqrt((sensitivity / 100) * (fpr / 100)) - (fpr / 100)) / ((sensitivity / 100) - (fpr / 100))
    return pt * 100  # for Percentage


def CSI(TP, FP, FN):
    # Threat Score (TS) or Critical Success Index (CSI)
    csi = TP / (TP + FN + FP)
    return csi * 100  # for Percentage


def BA(sensitivity, specificity):
    # Balanced Accuracy (BA)
    ba = (sensitivity + specificity) / 2
    return ba


def FM(sensitivity, precision):
    # Fowlkes–Mallows Index (FM)
    # FM = np.math.sqrt(precision * sensitivity)
    fm = np.math.sqrt((precision / 100) * (sensitivity / 100))
    return fm * 100


def BM(sensitivity, specificity):
    # Informedness or Bookmaker Informedness (BM)
    # BM = TPR + TNR - 1
    bm = ((sensitivity / 100) + (specificity / 100) - 1)
    return bm * 100


def MK(precision, npv):
    # Markedness (MK) or DeltaP (Δp)
    # MK = PPV + NPV - 1
    mk = ((precision / 100) + (npv / 100) - 1)
    return mk * 100


def PositiveLivelihoodRatio(tpr, fpr):
    # Positive Likelihood Ratio (LR+)
    lrplus = tpr / fpr
    return lrplus


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = fnr / tnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def Prevalence(TP, TN, FP, FN):
    # Prevalence = P / (P + N)
    prevalence = (TP + FN) / (TP + TN + FP + FN)
    return prevalence * 100


def ClassificationEvaluation(actual, predict):
    Max = preValidation(actual=actual, predict=predict)
    array, [TP, TN, FP, FN] = findConfusionMatrix(actual, predict, Max)
    accuracy = Accuracy(TP, TN, FP, FN)
    sensitivity = Sensitivity(TP, FN)
    specificity = Specificity(TN, FP)
    precision = Precision(TP, FP)
    fpr = FPR(TN, FP)
    fnr = FNR(TP, FN)
    npv = NPV(TN, FN)
    fdr = FDR(TP, FP)
    f1score = F1SCORE(TP, FP, FN)
    mcc = MCC(TP, TN, FP, FN)
    For = FOR(TN, FN)
    pt = PT(fpr, sensitivity)
    csi = CSI(TP, FP, FN)
    ba = BA(sensitivity, specificity)
    fm = FM(sensitivity, precision)
    bm = BM(sensitivity, specificity)
    mk = MK(precision, npv)
    lrplus = PositiveLivelihoodRatio(sensitivity, fpr)
    lrminus = NegativeLivelihoodRatio(specificity, fnr)
    dor = DOR(lrplus, lrminus)
    prevalence = Prevalence(TP, TN, FP, FN)
    Values = np.asarray([TP, TN, FP, FN, accuracy, sensitivity, specificity, precision, fpr, fnr, npv, fdr,
                         f1score, mcc, For, pt, csi, ba, fm, bm, mk, lrplus, lrminus, dor, prevalence])
    Verification(Values, array)
    return Values


def Verification(Values, array):
    Limit_0_100 = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    Limit_0_1 = np.array([13])
    if not (np.prod(array) == np.sum(Values[:4])):
        raise Exception("Something went wrong - Please check values")
    if (not (0 <= Values[Limit_0_100].all() <= 100)) or (not (0 <= Values[Limit_0_1].all() <= 1)):
        raise Exception('Something went wrong')


def Test():
    actual = np.zeros((10, 100, 200), dtype=np.uint8)
    pred = actual.copy()
    pred[0, :, :] = 1
    pred[1, :, :] = 1
    actual[0, :, :] = 1
    actual[5, :, :] = 1
    actual[7, :, :] = 1
    Values = ClassificationEvaluation(actual=actual, predict=pred)
    Table = PrettyTable()
    # Table.add_column("Measures", np.arange(len(Values)))
    Table.add_column("Values", Values)
    print(Table)
    # print(Values)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def net_evaluation(actual, predict):
    Max = preValidation(actual=actual, predict=predict)
    array, [TP, TN, FP, FN] = findConfusionMatrix(actual, predict, Max)
    Dice = (2 * TP) / ((2 * TP) + FP + FN) * 100
    Jaccard = TP / (TP + FP + FN) * 100  # IOU and Jac card are same metrics
    psnr = PSNR(actual, predict)
    mse = mean_squared_error(actual, predict)
    IoU = TP / (TP + FP + FN) * 100
    accuracy = Accuracy(TP, TN, FP, FN)
    sensitivity = Sensitivity(TP, FN)
    specificity = Specificity(TN, FP)
    precision = Precision(TP, FP)
    fpr = FPR(TN, FP)
    fnr = FNR(TP, FN)
    npv = NPV(TN, FN)
    fdr = FDR(TP, FP)
    f1score = F1SCORE(TP, FP, FN)
    mcc = MCC(TP, TN, FP, FN)
    RMSE = rmse(predict, actual)
    SSIM = ssim(predict, actual)
    MI = mutual_information_2d(predict.ravel(), actual.ravel())
    corr_coeff = np.corrcoef(predict.flatten(), actual.flatten())
    Correlation_Coefficient = np.mean(corr_coeff)
    # [SSIM[0], MI, RMSE, Correlation_Coefficient]
    EVAL = [TP, TN, FP, FN, Dice, Jaccard, accuracy, psnr, mse, sensitivity, specificity, precision, fpr, fnr, npv,
            fdr, f1score, mcc, SSIM[0]]
    return EVAL


def Evaluate_Image(Pred, Orig):
    RMSE = rmse(Pred, Orig)
    SSIM = ssim(Pred, Orig)
    MI = mutual_information_2d(Pred.ravel(), Orig.ravel())
    corr_coeff = np.corrcoef(Pred.flatten(), Orig.flatten())
    Correlation_Coefficient = np.mean(corr_coeff)
    EVAL = [SSIM[0], MI, RMSE, Correlation_Coefficient]
    return EVAL
