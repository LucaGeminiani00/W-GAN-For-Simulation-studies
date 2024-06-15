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