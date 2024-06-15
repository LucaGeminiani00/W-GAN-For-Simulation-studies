import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg

random_sample = df_generated.sample(n=1000,replace = True)
rd_sample = df.sample(n=1000,replace = True)
XG= random_sample["age"].values
YG = random_sample["re78"].values

rd_sample = df.sample(n=1000,replace = True)
X = rd_sample["age"].values
Y = rd_sample["re78"].values

# Perform kernel regression
kr2 = KernelReg(endog=Y, exog=X, var_type='c')
estimates2, _ = kr2.fit([i for i in range(int(min(X)), int(max(X))+1)])

# Perform kernel regression
kr = KernelReg(endog=YG, exog=XG, var_type='c')
estimates, _ = kr.fit([i for i in range(int(min(XG)), int(max(XG))+1)])

# Plot the kernel regression line
plt.plot([i for i in range(int(min(XG)), int(max(XG))+1)], estimates, color='red', label='Unpenalized W-GAN')
plt.plot([i for i in range(int(min(X)), int(max(X))+1)], estimates2, color='blue', label='Original')
plt.xlabel('Age')
plt.ylabel('Earnings 78 ')
plt.legend()
plt.title('Kernel Regression')
plt.show() 