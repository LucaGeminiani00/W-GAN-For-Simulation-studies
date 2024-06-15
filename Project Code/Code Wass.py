import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as D

# Setup the Data
file_path = "C:/Users/geminia/Desktop/ds-wgan-master/data/original_data/experimental.feather"
df: pd.DataFrame = pd.read_feather(file_path).drop(["u74", "u75"], axis=1)
df_balanced = df.sample(2 * len(df),weights=(1 - df.t.mean()) * df.t + df.t.mean() * (1 - df.t),replace=True,)  # balanced df for training
#Similar number of treated\ untreated units. 

# X | t
continuous_vars_0 = ["age", "education", "re74", "re75"]
continuous_lower_bounds_0 = {"re74": 0, "re75": 0}
categorical_vars_0 = ["black", "hispanic", "married", "nodegree"]
context_vars_0 = ["t"] #This is what you are conditioning on, that is in the first case only on treatment 

# Y | X, t
continuous_vars_1 = ["re78"] #This is your Y: Real earnings in 1978 
continuous_lower_bounds_1 = {"re78": 0}
categorical_vars_1 = [] #You don't want to generate any categorical variables in this case 
context_vars_1 = ["t", "age", "education", "re74", "re75", "black", "hispanic", "married", "nodegree"]

# Define the DataWrappers, the Specification, and the Generators/Critic classes 

# Initialize objects
data_wrappers = [DataWrapper(df_balanced, continuous_vars_0, categorical_vars_0,
                                  context_vars_0, continuous_lower_bounds_0),
                        DataWrapper(df_balanced, continuous_vars_1, categorical_vars_1,
                                  context_vars_1, continuous_lower_bounds_1)]
specs = [Specifications(dw, batch_size=128, max_epochs=1000, critic_lr=1e-3, generator_lr=1e-3,
                             print_every=100, device = "cpu") for dw in data_wrappers]
generators = [Generator(spec) for spec in specs]
critics = [Critic(spec) for spec in specs]

#Training: X | t 
x, context = data_wrappers[0].preprocess(df_balanced)  #What this is doing is simply standardizing the training Data.The categorical variables are one-hot encoded into 1s. 
train(generators[0], critics[0], x, context, specs[0])

#train(generators[0], critics[0], x, context, specs[0], monotonicity_penalty_chetverikov(5)) #Example of penalized training with Chetverikov

#Train Y | X, t
x, context = data_wrappers[1].preprocess(df_balanced)
train(generators[1], critics[1], x, context, specs[1])

#Simulate data with conditional WGANs
df_generated = data_wrappers[0].apply_generator(generators[0], df.sample(int(1e6), replace=True)) 
df_generated = data_wrappers[1].apply_generator(generators[1], df_generated)  #Condition Y on the generated covariates

#Add counterfactual outcomes 
from copy import copy

df_generated_cf = copy(df_generated)
df_generated_cf["t"] = 1 - df_generated_cf["t"]
df_generated["re78_cf"] = data_wrappers[1].apply_generator(generators[1], df_generated_cf)["re78"] 

#Computation of ATT 
att = ((df_generated.re78-df_generated.re78_cf) * (2*df_generated.t - 1))[df_generated.t==1].mean()
print(att)

compare_dfs(df, df_generated,scatterplot=dict(x=["re74", "age", "education"], y=["re78", "re75"], samples=400, smooth=0), 
            table_groupby=["t"], histogram=dict(variables=["re78", "re74", "age", "education"], nrow=2, ncol=2),figsize=3)

####Plot of conditional histogramS (74 = 0)
conditional_real = df[df['re74'] == 0]
conditional_generated = df_generated[df_generated['re74'] == 0]

conditional_real = df[df['re74'] > 0]
conditional_generated = df_generated[df_generated['re74'] > 0]

compare_dfs(conditional_real, conditional_generated,scatterplot=dict(x=["re74", "age", "education"], y=["re78", "re75"], samples=400, smooth=0), 
            table_groupby=["t"], histogram=dict(variables=["re78","re74", "age", "education"], nrow=2, ncol=2),figsize=3)

########################################

