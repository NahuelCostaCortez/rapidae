# Rapidae: Python Library for Rapid Creation and Experimentation of Autoencoders

[![Documentation Status](https://readthedocs.org/projects/rapidae/badge/?version=latest)](https://rapidae.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

🔗 [Documentation](https://rapidae.readthedocs.io/en/latest/) | 🔗 [PyPI Package](https://pypi.org/project/rapidae/)

## Description 📕

Rapidae is a Python library specialized in simplifying the creation and experimentation of autoencoder models. With a focus on ease of use, this library allows users to explore and develop autoencoder models in an efficient and straightforward manner.

I decided to develop this library to optimize my research workflow and provide a comprehensive resource for educators and learners exploring autoencoder models.

As a researcher, I often found myself spending time on repetitive tasks, such as creating project structures or replicating baseline models. (I've lost count of how many times I've gone through the excellent Keras VAE tutorial just to copy the model as a baseline for other experiments.)

As an educator, despite recognizing numerous fantastic online resources, I felt the need for a place where the features I consider important for teaching these models are consolidated: explanation, implementation, and versatility across different backends. The latter is particularly crucial, considering that PyTorch practitioners may find tedious to switch to TensorFlow, and vice versa. With the recently released Keras 3, Rapidae ensures that the user is met with a seamless and engaging experience, enabling to focus on model creation rather than backend specifics.

In summary, this library is designed to be simple enough for educational purposes, yet robust for researchers to concentrate on developing their models and conducting benchmark experiments in a unified environment.

🚨**Call for contributions**🚨

If you want to add your model to the package or collaborate in the package development feel free to shoot me a message at costanahuel@uniovi.es or just open an issue or a pull request. I´ll be happy to collaborate with you.

## Quick access:
- [Main Features](#main-features)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Switching backends](#switching-backends)
- [Available models](#available-models) 
- [Documentation](https://rapidae.readthedocs.io/en/latest/)
- [Citing this repository](#citation)


## Main features ⚙️

- **Ease of Use:** Rapidae has been designed to make the process of creating and experimenting with autoencoders as simple as possible, users can create and train autoencoder models with just a few lines of code.

- **Backend versatility:** Rapidae relies on Keras 3.0, which is backend agnostic, allowing switching indistinctly between Tensorflow, Pytorch or Jax.

- **Customization:** Easily customize model architecture, loss functions, and training parameters to suit your specific use case.

- **Experimentation:** Conduct experiments with different hyperparameters and configurations to optimize the performance of your models.

## Overview 📦	

Rapidae is structured as follows:

- **data:** This module contains everything related to the acquisition and preprocessing of datasets. 

- **models:** This is the core module of the library. It includes the base architectures on which new ones can be created, several predefined architectures and a list of predefined default encoders and decoders.

- **pipelines:** Pipelines are designed to perform a specific task or set of tasks such as data preprocessing or model training.

- **evaluate:** Its main functionality is the evaluation of model performance. It also includes a tool utils for various tasks: latent space visualization, reconstructions, evaluation, etc.
 

## Installation ⚙️

The library has been tested with Python versions <3.12, >=3.10, therefore we recommend first creating a **virtual environment** with a suitable python version. Here´s an example with conda:

```conda create -n rapidae python=3.10.13```

Then, just activate the environment with ```conda activate rapidae``` and install the library.

>[!NOTE]
>If you are using Google Colab, you are good to go (i.e. you do not need to create an environment). The library is fully compatible with Colab´s default environment.


### With Pip
To install the latest stable release of this library run the following:

```bash
  pip install rapidae
```

Note that you will also need to install a backend framework. Here are the official installation guidelines:

- [Installing TensorFlow](https://www.tensorflow.org/install)
- [Installing PyTorch](https://pytorch.org/get-started/locally/)
- [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html)

>[!IMPORTANT]
>If you install TensorFlow, you should reinstall Keras 3 afterwards via ```pip install --upgrade keras```. This is a temporary step while TensorFlow is pinned to Keras 2, and will no longer be necessary after TensorFlow 2.16. The cause is that tensorflow==2.15 will overwrite your Keras installation with keras==2.15.

### From source code
You can also clone the repo to have fully access to all the code. Some features may not yet be available in the published stable version so this is the best way to stay up-to-date with the latest updates.

```bash
git clone https://github.com/NahuelCostaCortez/rapidae
cd rapidae
```

Then you only have to install the requirements:

```bash
pip install -r requirements.txt
```

## Usage 🫳🏻
[Here](https://github.com/NahuelCostaCortez/rapidae/blob/main/examples/00Quickstart_tutorial.ipynb) you have a simple tutorial with the most relevant aspects of the library. In addition, in the [examples folder](https://github.com/NahuelCostaCortez/rapidae/tree/main/examples), you will find a series of notebooks for each model and with particular use cases.

You can also use a web interface made with Streamlit where you can load datasets, configure models and hypeparameters, train, and evaluate the results. Check the [web interface](https://github.com/NahuelCostaCortez/rapidae/blob/main/examples/web_interface.ipynb) notebook.

## Switching backends 💻
Since Rapidae uses Keras 3, you can easily switch among Tensorflow, Pytorch and Jax (Tensorflow is the selected option by default).

You can export the environment variable KERAS_BACKEND or you can edit your local config file at ~/.keras/keras.json to configure your backend. Available backend options are: "jax", "tensorflow", "torch". Example:

```bash
export KERAS_BACKEND="torch"
```

In a notebook, you can do:

```c
import os
os.environ["KERAS_BACKEND"] = "torch" 
import keras
```

## Available Models 🚀

Below is the list of the models currently implemented in the library.

|               Models               |                                                                                    Training example                                                                                    |                     Paper                    |                           Official Implementation                          |
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------------------------------------:|
| Autoencoder (AE)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/QUICKSTART_tutorial.ipynb) | [link](https://www.science.org/doi/abs/10.1126/science.1127647)                                             |  
| Denoising Autoencoder                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/denoising_autoencoder.ipynb) | [link](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)                                             | [link](https://keras.io/examples/vision/autoencoder/)
| Sparse Autoencoder                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/sparse_autoencoder.ipynb) | [link](https://arxiv.org/abs/1312.5663)                                             |   
| Contractive Autoencoder                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/contractive_autoencoder.ipynb) | [link](http://www.icml-2011.org/papers/455_icmlpaper.pdf)                                             |  
| Variational Autoencoder (VAE)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae.ipynb) | [link](https://arxiv.org/abs/1312.6114)                                             | [link](https://keras.io/examples/generative/vae/)
| Beta Variational Autoencoder (BetaVAE)                 | TO-DO | [link](https://openreview.net/pdf?id=Sy2fzU9gl)                                             | 
| Vector Quantised-Variational AutoEncoder (VQ-VAE)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/vae.ipynb) | [link](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)                                             | [link](https://keras.io/examples/generative/vq_vae/)
| Recurrent Variational Encoder (RVE)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/rapidae/blob/main/examples/RVE_RUL.ipynb) | [link](https://www.sciencedirect.com/science/article/pii/S0951832022000321)                                             | [link](https://github.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational)
| ICFormer                 | TO-DO | [link](https://www.sciencedirect.com/science/article/pii/S0378775323012867)                                             | [link](https://github.com/NahuelCostaCortez/ICFormer)
| Hierarchical Variational Autoencoder (HVAE)                 | TO-DO | [link](https://arxiv.org/abs/1905.06845)                                             | [link](https://github.com/fhkingma/bitswap)
| interval-valued Variational Autoencoder (iVAE)                 | IN PROGRESS |                                             | 


## Documentation 📚

Check out the full documentation for detailed information on installation, usage, examples and recipes: 🔗 [Documentation Link](https://rapidae.readthedocs.io/en/latest/)

All documentation source and configuration files are located inside the docs directory.


## Dealing with issues 🛠️	

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/NahuelCostaCortez/rapidae/issues).


## Citation ✒️

If you find this work useful or incorporate it into your research, please consider citing it. You will find the citation information in the "Cite this repository" button below the "About" section.
