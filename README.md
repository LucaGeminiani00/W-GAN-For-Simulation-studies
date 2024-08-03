# W-GAN-For-Simulation-studies
This repository proposes a PyTorch implementation of W-GAN with Penalized Gradients (Gulrajani et al. https://arxiv.org/abs/1704.00028) for generating artificial cross-sectional data. 

Code can be easily run on CPU if using the LDW-E dataset, whilst a GPU (easy implementation in Colab) should be used in case of LDW-CPS and LDW-PSID. An explanation of how to run the code can be found in the RunWGAN.py file. 

The specific application is to economic datasets, though it may be easily extended to other domains. Further extensions of such algorithms might involve time series and panel datasets. 
Several chunks of code are taken from the implementation proposed in the paper "Using Wasserstein GAN for the design of Monte Carlo simulations", published by Guido Imbens et al. https://arxiv.org/abs/1909.02210 \.
