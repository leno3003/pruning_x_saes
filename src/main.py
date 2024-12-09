# -*- coding: utf-8 -*-

import os
import sys
from tkinter import W

# from torchsummary import summary

base_artifacts_path = "/scratch/pruning_x_saes_artifacts/"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
root = "/content"

import gc
from toy_model import ToyModel, ToyModelConfig
from prune_utils import compute_sparsity_global, prune_model, compute_sparsity_toy
import itertools
import math
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch as t

# from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, clear_output, display
# from openai import OpenAI
from rich import print as rprint
from rich.table import Table

from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part32_interp_with_saes").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import utils as part31_utils

NUMBER_OF_INSTANCES = 8
NUMBER_OF_INPUT_FEATURES = 5
NUMBER_OF_HIDDEN_UNITS = 2
TRAINING_STEPS = 10_000
SPARSITY_THRESHOLD = 90


MAIN = __name__ == "__main__"


cfg = ToyModelConfig(n_inst=NUMBER_OF_INSTANCES, n_features=NUMBER_OF_INPUT_FEATURES, d_hidden=NUMBER_OF_HIDDEN_UNITS)

importance = 0.9 ** t.arange(cfg.n_features)
feature_probability = 50 ** -t.linspace(0, 1, cfg.n_inst)

print(feature_probability)


model = ToyModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)


sparsity = compute_sparsity_toy(model)
print(sparsity)

pruning_iteration = 0

while sparsity < SPARSITY_THRESHOLD:

    model.optimize(steps=TRAINING_STEPS)
    
    directory = base_artifacts_path + f'{NUMBER_OF_INSTANCES}_instances/'+ f'pruning_iteration_{pruning_iteration}/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if NUMBER_OF_INSTANCES == 1:
        part31_utils.plot_features_in_2d(
            model.W,
            colors=model.importance,
            title=f"Superposition: {cfg.n_features} features represented in 2D space",
            subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability],
            path= directory + f'2D_toy_model_superposition_{sparsity}.webp',
        )

    else:
        part31_utils.plot_features_in_2d(
            model.W,
            colors=model.importance,
            title=f"Superposition: {cfg.n_features} features represented in 2D space",
            subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
            path= directory + f'2D_toy_model_superposition_{sparsity}.webp',
        )
    
    with t.inference_mode():
        batch = model.generate_batch(200)
        hidden = einops.einsum(
            batch,
            model.W,
            "batch_size instances features, instances hidden features -> instances hidden batch_size",
        )
    
    if NUMBER_OF_HIDDEN_UNITS == 2:
        part31_utils.plot_features_in_2d(hidden, title="Hidden state representation of a random batch of data", path= directory + f'2D_toy_model_hidden_{sparsity}.webp')
    
    else:
        part31_utils.plot_features_in_Nd(
            model.W,
            height=800,
            width=1600,
            title="ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
            subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability],
            path = directory + f'ND_toy_model_hidden_{sparsity}.webp',
        )
    
    model = prune_model(model)

    sparsity = compute_sparsity_toy(model)
    pruning_iteration += 1

    print(f'Sparsity at pruning iteration {pruning_iteration}: {sparsity}')


print('Done.')