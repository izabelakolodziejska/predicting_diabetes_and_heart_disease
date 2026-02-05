import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multi_model import multi_model
from src.metrics import find_optimal_threshold, calculate_metrics

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import torch
import scipy
import json



def train_multi():

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')

    y_diabetes_train = pd.read_csv(f'data/processed/y_diabetes_train.csv')
    y_diabetes_val = pd.read_csv(f'data/processed/y_diabetes_val.csv')
    y_diabetes_test = pd.read_csv(f'data/processed/y_diabetes_test.csv')

    y_heart_train = pd.read_csv(f'data/processed/y_heart_train.csv')
    y_heart_val = pd.read_csv(f'data/processed/y_heart_val.csv')
    y_heart_test = pd.read_csv(f'data/processed/y_heart_test.csv')

    #testy
    """ 
    X_train = X_train.head(100)
    X_val = X_val.head(100)
    X_test = X_test.head(100)
    y_diabetes_train = y_diabetes_train.head(100)
    y_diabetes_val = y_diabetes_val.head(100)
    y_diabetes_test = y_diabetes_test.head(100)
    y_heart_train = y_heart_train.head(100)
    y_heart_val = y_heart_val.head(100)
    y_heart_test = y_heart_test.head(100)  """

    clf = multi_model(X_train)


    X_train = scipy.sparse.csr_matrix(X_train.values)
    X_val = scipy.sparse.csr_matrix(X_val.values)
    X_test = X_test.to_numpy(dtype=np.float64)

    y_train = np.column_stack([y_diabetes_train.to_numpy().ravel(), y_heart_train.to_numpy().ravel()])
    y_val = np.column_stack([y_diabetes_val.to_numpy().ravel(), y_heart_val.to_numpy().ravel()])
    y_test = np.column_stack([y_diabetes_test.to_numpy().ravel(), y_heart_test.to_numpy().ravel()])

    max_epochs = 60


    

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        max_epochs=max_epochs,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        loss_fn=[torch.nn.functional.cross_entropy]*2

    )

    y_pred_proba = clf.predict_proba(X_test)
    y_diabetes_pred_proba = y_pred_proba[0][:, 1]
    y_heart_pred_proba = y_pred_proba[1][:, 1]
    #preds = clf.predict(X_test)

    diabetes_optimal_thresh = find_optimal_threshold(
        y_diabetes_test, y_diabetes_pred_proba, metric='f1'
    )
    diabetes_metrics = calculate_metrics(
        y_diabetes_test, y_diabetes_pred_proba, threshold=diabetes_optimal_thresh
    )

    heart_optimal_thresh = find_optimal_threshold(
        y_heart_test, y_heart_pred_proba, metric='f1'
    )
    heart_metrics = calculate_metrics(
        y_heart_test, y_heart_pred_proba, threshold=heart_optimal_thresh
    )

    #wynik auc dla kazdego taska
    test_aucs = [
        roc_auc_score(y_score=y_pred_proba[0][:, 1], y_true=y_test[:, 0]), 
        roc_auc_score(y_score=y_pred_proba[1][:, 1], y_true=y_test[:, 1])  
    ]
    
    ensemble_pred = np.mean(np.vstack([task_pred[:, 1] for task_pred in y_pred_proba]), axis=0)
    ensemble_auc = roc_auc_score(y_score=ensemble_pred, y_true=y_test[:, 0])

    results = {
        "diabetes": {
            "metrics": diabetes_metrics 
        },
        "heart": {
            "metrics": heart_metrics
        },
        "test_aucs": [float(auc) for auc in test_aucs],
        "ensemble_auc": float(ensemble_auc)
    }
    
    dir = 'results/multi_task'
    os.makedirs(dir, exist_ok=True)

    with open(f'{dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    clf.save_model(str(f"{dir}/multi_task"))

    print("Training ended, saved results")

if __name__ == '__main__':
    train_multi()