import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Just run this code 
# Sample data
np.random.seed(42)
df_real = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(5, 2, 100),
    'C': np.random.normal(-3, 1, 100)
})

df_gen = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(5, 2, 100),
    'C': np.random.normal(-3, 1, 100)
})

# Calculate correlation matrices
corr_real = df_real.corr()
corr_generated = df_gen.corr()

# Plotting the heatmap for the real dataset
plt.figure(figsize=(10, 8))
sns.heatmap(corr_real, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Real Dataset')
plt.show()

# Plotting the heatmap for the generated dataset
plt.figure(figsize=(10, 8))
sns.heatmap(corr_generated, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Generated Dataset')
plt.show()