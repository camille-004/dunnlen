import numpy as np


def get_tp_fp_tn_fn(labels_train, labels_predicted) -> object:
    tp = np.sum((labels_train == 1) & (labels_predicted == 1))
    fp = np.sum((labels_train == 0) & (labels_predicted == 1))
    tn = np.sum((labels_train == 0) & (labels_predicted == 0))
    fn = np.sum((labels_train == 1) & (labels_predicted == 0))
    return tp, fp, tn, fn


def accuracy(labels_train, labels_predicted):
    return np.sum(labels_train == labels_predicted) / len(labels_train)


def precision(labels_train, labels_predicted):
    tp, fp, tn, fn = get_tp_fp_tn_fn(labels_train, labels_predicted)
    return tp / (tp + fp)


def recall(labels_train, labels_predicted):
    tp, fp, tn, fn = get_tp_fp_tn_fn(labels_train, labels_predicted)
    return tp / (fn + tp)


def f1(labels_train, labels_predicted):
    pr = precision(labels_train, labels_predicted)
    re = recall(labels_train, labels_predicted)
    return (2 * pr * re) / (pr + re)


def return_metrics(labels_train, labels_predicted):
    ac = accuracy(labels_train, labels_predicted)
    pr = precision(labels_train, labels_predicted)
    re = recall(labels_train, labels_predicted)
    f1_ = f1(labels_train, labels_predicted)
    return ac, pr, re, f1_


def print_metrics(labels_train, labels_predicted):
    print(f'Accuracy: {accuracy(labels_train, labels_predicted)}')
    print(f'Precision: {precision(labels_train, labels_predicted)}')
    print(f'Recall: {recall(labels_train, labels_predicted)}')
    print(f'F1: {f1(labels_train, labels_predicted)}')
