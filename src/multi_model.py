import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch
from src.metrics import load_metadata_pickle

def multi_model(X_train):

    cat_idxs, cat_dims = load_metadata_pickle()

    clf = TabNetMultiTaskClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        n_d=8,
        n_a=8,
        n_steps=5,
        gamma=1.3, 
        n_shared=2,
        n_independent=2, 
        lambda_sparse=1e-4,
        seed = 0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax' #this is the masking function to use for selecting features.
    )

    return clf
