import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))

    clf = fit_clf(train_repr, train_labels)
    y_pred = clf.predict(test_repr)
    y_score = clf.predict_proba(test_repr) if eval_protocol == 'linear' else clf.decision_function(test_repr)

    metrics = {
        'acc': clf.score(test_repr, test_labels),
        "f1": f1_score(test_labels, y_pred, average='macro', zero_division=np.nan),
        "precision": precision_score(test_labels, y_pred, average='macro', zero_division=np.nan),
        "recall": recall_score(test_labels, y_pred, average='macro', zero_division=np.nan),
        'auprc': average_precision_score(test_labels_onehot, y_score),
    }
    
    return y_score, metrics
