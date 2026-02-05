import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    #zmiana prawdopodobieństwa na tak/nie według threshold
    y_pred = (y_pred_proba > threshold).astype(int)

    metrics = {
        'accuracy' : accuracy_score(y_true, y_pred), #True Positives + True Negatives / wszystko, im wyżej tym, lepiej chociaż w przypadku tego modelu mniej pomocne bo duża dysproporcja klas
        'precision' : precision_score(y_true, y_pred, zero_division=0), #True Positives / (True Positives + False Positives), ze względu na małą ilość pozytywów mniej istotny
        'recall' : recall_score(y_true, y_pred, zero_division=0), #True Positives / (True Positives + False Negatives)
        'f1' : f1_score(y_true, y_pred, zero_division=0), #F1 = 2 * (Precision * Recall) / (Precision + Recall), balans między precision a recall
        'auc' : roc_auc_score(y_true, y_pred_proba), #jak dobrze model rozróżnia klasy
        'confusion_matrix' : confusion_matrix(y_true, y_pred).tolist()
    }

    return metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    #threshold decyduje kiedy uznajemy kogoś za chorego (przy jakim prawdopodobieństwie)
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"\nOptimal threshold for {metric}: {optimal_threshold:.2f}")
    print(f"Best {metric} score: {best_score:.4f}")
    
    return optimal_threshold

def calculate_class_weights(y_train):
    # nadaje większą wagę przypadkom choroby, pomocne ponieważ jest dużo mniej chorych, w MTL opcja niedostępna
    y_train = y_train.values.flatten() 
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[int(class_label)] = class_weight 
    
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    print(f"Class weights: {class_weights}")

    return class_weights


def load_metadata_pickle():
    with open('data/metadata/metadata.pkl', 'rb') as f:
        cat_idxs, cat_dims = pickle.load(f)
    return cat_idxs, cat_dims