# W-GAN-For-Simulation-studies
This repository proposes a PyTorch implementation of W-GAN with Penalized Gradients (Gulrajani et al.) for generating artificial cross-sectional data. Code can be run on CPU concerning LDW-E, whilst a GPU (easy implementation in Colab) should be used in case of LDW-CPS and LDW-PSID. 
The specific application is to economic datasets, though it may be easily extended to other domains. Further extensions of such algorithms might involve time series and panel datasets. \
Large chunks of code are taken from the implementation proposed in the paper "Using Wasserstein GAN for the design of Monte Carlo simulations", published by Guido Imbens et al. https://arxiv.org/abs/1909.02210 \
This repository is self-contained, though it is not in optimal form concerning code and project structure. 
