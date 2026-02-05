import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.single_model import single_model
from src.metrics import find_optimal_threshold, calculate_metrics, calculate_class_weights

import pandas as pd
import numpy as np
import scipy
import json
from pytorch_tabnet.augmentations import ClassificationSMOTE



def train_single_task_model(task="diabetes"):

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')

    y_train = pd.read_csv(f'data/processed/y_{task}_train.csv')
    y_val = pd.read_csv(f'data/processed/y_{task}_val.csv')
    y_test = pd.read_csv(f'data/processed/y_{task}_test.csv')

    #do testów
    """ X_train = X_train.head(100)
    X_val = X_val.head(100)
    X_test = X_test.head(100)
    y_train = y_train.head(100)
    y_val = y_val.head(100)
    y_test = y_test.head(100)   """

    max_epochs = 50

    aug = ClassificationSMOTE(p=0.2)

    clf = single_model(X_train)

    class_weights = calculate_class_weights(y_train)

    X_train = scipy.sparse.csr_matrix(X_train.values)
    X_val = scipy.sparse.csr_matrix(X_val.values)
    X_test = X_test.to_numpy(dtype=np.float64)
    y_train = y_train.values.flatten()
    y_val = y_val.values.flatten()
    y_test = y_test.values.flatten()

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], #List of eval tuple set (X, y).The last one is used for early stopping
        eval_name=['train', 'val'],
        eval_metric=['auc'], 
        max_epochs=max_epochs,
        patience=10, #Number of consecutive epochs without improvement before performing early stopping.
        weights=class_weights,
        batch_size=16384, #Number of examples per batch. Large batch sizes are recommended.
        virtual_batch_size=256, #Size of the mini batches used for “Ghost Batch Normalization”
        augmentations=aug
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    #preds = clf.predict(X_test)

    optimal_f1 = find_optimal_threshold(y_test, y_pred_proba, metric='f1')
    metrics_f1 = calculate_metrics(y_test, y_pred_proba, threshold=optimal_f1)


    
    dir = f'results/single_task_{task}'
    os.makedirs(f'{dir}', exist_ok=True)

    with open(f'{dir}/results.json', 'w') as f:
        json.dump(metrics_f1, f, ensure_ascii=False, indent=2)
    print("Saved metrics to results.json")

    clf.save_model(str(f"{dir}/single_{task}"))

    print("Training ended, saved results")

if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'diabetes'
    train_single_task_model(task)