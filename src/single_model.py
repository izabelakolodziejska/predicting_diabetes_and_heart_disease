import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_tabnet.tab_model import TabNetClassifier
from src.metrics import load_metadata_pickle
import torch

def single_model(X_train):

    cat_idxs, cat_dims = load_metadata_pickle()


    clf = TabNetClassifier(
        n_d=64, #Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting. Values typically range from 8 to 64.
        n_a=64, #Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice. (default=8)
        n_steps=5, #Number of steps in the architecture (usually between 3 and 10)
        gamma=1.5, #This is the coefficient for feature reusage in the masks. A value close to 1 will make mask selection least correlated between layers. Values range from 1.0 to 2.0.
        cat_idxs=cat_idxs, #List of categorical features indices.
        cat_dims=cat_dims, #List of categorical features number of modalities (number of unique values for a categorical feature)
        cat_emb_dim=1, #List of embeddings size for each categorical features. (default =1)
        n_independent=2, #Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.
        n_shared=2, #Number of shared Gated Linear Units at each step Usual values range from 1 to 5
        epsilon=1e-15,
        seed=0,
        momentum=0.3, #Momentum for batch normalization, typically ranges from 0.01 to 0.4
        clip_value=2., #If a float is given this will clip the gradient at clip_value.
        lambda_sparse=1e-4, #This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2), #Parameters compatible with optimizer_fn used initialize the optimizer. Since we have Adam as our default optimizer, we use this to define the initial learning rate used for training. As mentionned in the original paper, a large initial learning rate of 0.02 with decay is a good option.
        scheduler_params = {"gamma": 0.95,
                     "step_size": 20},  #Dictionnary of parameters to apply to the scheduler_fn. Ex : {”gamma”: 0.95, “step_size”: 10}
        scheduler_fn=torch.optim.lr_scheduler.StepLR #Pytorch Scheduler to change learning rates during training.
    )

    return clf

