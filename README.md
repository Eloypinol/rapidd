# Project name here
> Summary description here.


```python
from ebm.models import RBM
```

```python
from rapidd.core import *
```

This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

```python
1+1
```




    2



```python
say_hello("Eloy")
```




    'Hello Eloy!'



## Aqui empieza Comparison

```python
import torch
from itertools import product
```

```python
#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd     = 1000        # Number of nodes in the hidden layer
rapid_lr = 0.02        # Learning rate for RAPID
epochs   = 300         # Training epochs
K        = 8           # Number of patterns for RA model
bs_rapid = 3           # Batch size for RAPID
gpu      = False        # Use of GPU
```

```python
#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------
# =============================================================================
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

all_datasets = create_bars_4x4()
for _ in all_datasets:
    dataset = all_datasets.pop(0)
    dataset = dataset.to(device)
    all_datasets.append(dataset)

train_set, recon_train, test_set, recon_test = all_datasets

vis = len(train_set[0])

all_confs = torch.Tensor(list(product([-1, 1], repeat=vis))).to(device)


# Construct Models

opt_ra  = SGD_xi(rapid_lr)

rbm_ra = RA_RBM(n_visible=vis,
                n_hidden=hidd,
                K=K,
                optimizer=opt_ra,
                device=device
                ).to(device)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

machines = [rbm_ra]

for epoch in range(epochs):
    train_loader_rapid = torch.utils.data.DataLoader(train_set,
                                                     batch_size=bs_rapid,
                                                     shuffle=True)


    rbm_ra.train(train_loader_rapid)
```
