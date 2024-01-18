# Examples

## Overview 

On this page, you'll find a step-by-step walkthrough of building the different autoencoder models available in Rapidae for a specific use case.  
The code for the examples can be found inside the examples directory as Jupyter Notebooks.

Usually these example covers:

- Data preparation: Guidance on obtaining and preparing the dataset for training the autoencoder. It usually contains different preprocessing and traint/test/val separation.

- Model architecture: Code snippets illustrating the architecture of the autoencoder model.

- Training: Instructions on how to create a training pipeline to train the model, including hyperparameters and training options.

- Evaluation: Tips on evaluating the performance of the trained model over the test set, including some kind of visualization of reconstructed outputs or latent space.

## Denoising Autoencoder

Unlike traditional autoencoders that aim to reconstruct the input from a compressed representation, denoising autoencoders are trained to reconstruct clean, uncorrupted data from noisy or partially obscured input. The training process involves introducing noise or distortion to the input data and then training the autoencoder to recover the original, clean data. This approach encourages the model to learn robust features that capture the essential information needed to reconstruct the clean input, making it less sensitive to irrelevant or noisy details. Denosing autoencoders find applications in various areas, including image denoising, data compression, and feature learning, where the goal is to extract meaningful representations from noisy or imperfect data.

Architecture: Dense layers

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/denoising_autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Sparse Autoencoder

Unlike traditional autoencoders, which aim to replicate the input data, sparse autoencoders incorporate a sparsity constraint, encouraging the network to activate only a small number of neurons in the hidden layer. This sparsity constraint helps extract meaningful features and representations from the input data, making the network more robust and efficient in capturing essential patterns. Sparse autoencoders find applications in various domains, such as image and signal processing, where learning compact and meaningful representations is crucial for tasks like classification and reconstruction.

Architecture: Dense layers

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/sparse_autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Contractive Autoencoder

Contractive autoencoders incorporate a regularization term during training that penalizes changes in the hidden layer activations concerning the input data. This regularization encourages the autoencoder to learn robust and stable features by limiting the sensitivity of the hidden layer to small changes in the input. The contractive autoencoder is particularly useful for capturing essential patterns and reducing the impact of irrelevant variations in the input data. This makes it suitable for applications where extracting meaningful and stable representations is crucial, such as in denoising or dimensionality reduction tasks.

Architecture: Dense layers

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/contractive_autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## VAE

A Variational Autoencoder (VAE) is a type of generative model that combines the principles of autoencoders and probabilistic modeling. VAEs are designed to learn a probabilistic mapping between high-dimensional input data and a lower-dimensional latent space, capturing meaningful representations of the input data. In a VAE, the encoder network maps input data to a distribution in the latent space, typically modeled as a Gaussian distribution. The decoder network then samples from this distribution to reconstruct the input data. Importantly, VAEs introduce a probabilistic element by enforcing that the latent space follows a specific probability distribution, usually a multivariate Gaussian. During training, VAEs maximize a variational lower bound on the log-likelihood of the data. This involves minimizing the reconstruction error, ensuring that the generated samples resemble the input data, and regularizing the distribution of the latent space to follow the desired probability distribution.

VAEs have applications in generative tasks, such as image and text generation, and are valued for their ability to generate diverse and realistic samples while providing a structured latent space that allows for interpolation and manipulation of data representations.

Architecture: MLP

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Vector Quantised-VAE

A Vector Quantized-Variational Autoencoder (VQ-VAE) is an extension of the traditional Variational Autoencoder (VAE) that incorporates discrete latent variables. In VQ-VAE, the encoder maps the input data to continuous latent variables, and a discrete codebook is introduced to quantize these continuous variables. This quantization helps to introduce a form of discrete structure into the latent space, making it more interpretable and potentially capturing more meaningful representations.

The key components of a VQ-VAE include:

1. **Encoder:** This part of the network maps the input data to continuous latent variables.

2. **Codebook:** A set of discrete vectors that serve as the representative centroids of the quantized latent space.

3. **Quantization:** The continuous latent variables from the encoder are quantized by finding the closest vector in the codebook. This discrete representation is then used in the subsequent stages.

4. **Decoder:** The decoder reconstructs the input data using both the quantized discrete latent variables and the continuous variables.

VQ-VAEs are particularly useful when dealing with data that exhibits a mix of continuous and discrete patterns. They find applications in generative tasks where having a structured and interpretable latent space is essential, such as in generating diverse and meaningful samples in image and speech synthesis.

Architecture: Convolutional layers

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vq_vae.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Convolutional VAE for Image classification

A VAE for Image classification

Architecture: Convolutional layers, same as the one used in the  <a href="https://keras.io/examples/generative/vae/" target="_parent">keras tutorial</a>

Dataset: MNIST

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae_classifier_conv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Recurrent VAE for Atrial Fibrillation classification

A VAE for time series classification

Architecture: Bidirectional LSTMs

Dataset: Arrythmia - synthetic data from pacemakers logs

Original paper: <a href="https://ieeexplore.ieee.org/document/9373315" target="_parent">Semi-Supervised Recurrent Variational Autoencoder Approach for Visual Diagnosis of Atrial Fibrillation</a>

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae_classifier_rnn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Recurrent VAE for RUL estimation

A VAE based method for RUL estimation. The model consists only of an encoder and a decoder and it is optimized to minimize the KL loss and the regression loss at the same time, thus achieving a visual explanatory map in the latent space.

Architecture: Bidirectional LSTMs

Dataset: CMAPSS

Original paper: <a href="https://www.sciencedirect.com/science/article/pii/S0951832022000321" target="_parent">Variational encoding approach for interpretable assessment of remaining useful life estimation</a>

<a href="https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae_regressor_rnn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



