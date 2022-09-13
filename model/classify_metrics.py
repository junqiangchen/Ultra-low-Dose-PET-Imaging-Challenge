import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import auc
import seaborn as sns
from numpy.random import randint


# classificaiton metric
def accuracy_score(y_test, y_test_pred):
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    return accuracy


def Confidence_Intervals_accuracy_score(y_test, y_test_pred, Confidence_level=95, n_iterations=100):
    """
    Confidence_Intervals calculate,frome here:https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    :param y_test:
    :param y_test_pred:
    :param Confidence_level:
    :return:
    """
    # bootstrap
    num_samples = len(y_test)
    scores = list()
    for _ in range(n_iterations):
        # bootstrap sample
        indices = randint(0, num_samples, num_samples // 2)
        y_test_sample = y_test[indices]
        y_test_pred_sample = y_test_pred[indices]
        # calculate and store statistic
        statistic = accuracy_score(y_test_sample, y_test_pred_sample)
        scores.append(statistic)
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 100 - Confidence_level
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    CI_lower = max(0.0, np.percentile(scores, lower_p))
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    CI_upper = min(1.0, np.percentile(scores, upper_p))
    return CI_lower, CI_upper


def classification_reports(y_true, y_pred):
    print("classification_report(left:labels):")
    print(classification_report(y_true=y_true, y_pred=y_pred))


def confusion_matrixs(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("confusion _matrixs(left labels:y_true,up labels:y_pred):")
    print(conf_mat)
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, ax=ax)
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()


# ROC Curves summarize the trade-off between the true positive rate and
# false positive rate for a predictive model using different probability thresholds.
# Precision-Recall curves summarize the trade-off between the true positive rate and
# the positive predictive value for a predictive model using different probability thresholds.
# ROC curves are appropriate when the observations are balanced between each class,
# whereas precision-recall curves are appropriate for imbalanced datasets.
def roc_auc_scores(y_trueinput, y_pred_probinput, labelnumber):
    color = ['r--', 'g--', 'b--']
    roc_value_mean = []
    for i in range(labelnumber):
        # calculate roc curves
        y_true = y_trueinput.copy()
        y_pred_prob = y_pred_probinput[:, i].copy()
        y_true[y_trueinput == i] = 1
        y_true[y_trueinput != i] = 0
        fpr, tpr, th = roc_curve(y_true, y_pred_prob)  # 测试集
        roc_value = roc_auc_score(y_true=y_true, y_score=y_pred_prob)
        roc_value_mean.append(roc_value)
        # plot the roc curve for the model
        plt.plot(fpr, tpr, color[i], label=str(i) + ',' + f'AUC={round(auc(fpr, tpr), 4)}')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.title('ROC')
    plt.show()
    # calculate mean scores
    return np.mean(np.array(roc_value_mean))


def Precision_Recall_score(y_trueinput, y_pred_probinput, labelnumber):
    roc_value_mean = []
    color = ['r--', 'g--', 'b--']
    for i in range(labelnumber):
        # calculate roc curves
        y_true = y_trueinput.copy()
        y_pred_prob = y_pred_probinput[:, i].copy()
        y_true[y_trueinput == i] = 1
        y_true[y_trueinput != i] = 0
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred_prob)
        roc_value = auc(lr_recall, lr_precision)
        roc_value_mean.append(roc_value)
        # plot the precision-recall curves
        plt.plot(lr_recall, lr_precision, color[i],
                 label=str(i) + ',' + f'AUC={round(auc(lr_recall, lr_precision), 4)}')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.title('Precision_Recall')
    plt.show()
    return np.mean(np.array(roc_value_mean))


def classify_metric_message(predict_label_file, labelnumber=3):
    """
    :param predict_label_file:should have type:predict_label,predict_prob,true_label
    :param name:
    :return:
    """
    csvimagedata = pd.read_csv(predict_label_file)
    data = csvimagedata.iloc[:, :].values
    predict_labels = data[:, 0]
    true_labels = data[:, 1]
    predict_probs = data[:, 2:2 + labelnumber]

    print("roc_auc_score")
    print(roc_auc_scores(true_labels, predict_probs, labelnumber))
    print("Precision_Recall_score")
    print(Precision_Recall_score(true_labels, predict_probs, labelnumber))
    print("confusion_matric")
    confusion_matrixs(true_labels, predict_labels)
    print("main_classify")
    classification_reports(true_labels, predict_labels)


if __name__ == '__main__':
    predict_label_file = "classify_metrics_taskC.csv"
    classify_metric_message(predict_label_file, 3)
