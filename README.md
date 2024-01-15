# Rapidae: Python Library for Rapid Creation and Experimentation of Autoencoders

[![Documentation Status](https://readthedocs.org/projects/rapidae/badge/?version=latest)](https://rapidae.readthedocs.io/en/latest/?badge=latest)

🔗 [Documentation](https://rapidae.readthedocs.io/en/latest/) | 🔗 [PyPI Package](https://pypi.org/project/rapidae/)

## Description 💻

Rapidae is a Python library specialized in simplifying the creation and experimentation of autoencoder models. With a focus on ease of use, Rapidae allows users to explore and develop autoencoder models in an efficient and straightforward manner.

## Main features 💪

- **Ease of Use:** Rapidae has been designed to make the process of creating and experimenting with autoencoders as simple as possible, users can create and train autoencoder models with just a few lines of code.

- **Backend versatility:** Rapidae was develovep using the new version Keras 3.0. This adds the ability to run experiments on three different backends (Tensorflow, Pytorch and Jax) allows users to take advantage of the specific strengths of each without having to learn new syntaxes. Rapidae handles the abstraction, allowing researchers to focus on the design and evaluation of their models.

- **Customization:** Easily customize model architecture, loss functions, and training parameters to suit your specific use case.

- **Experimentation:** Conduct experiments with different hyperparameters and configurations to optimize the performance of your autoencoder models.

## Overview 📦	

Rapidae is structured as follows:

- **data:** This module contains everything related to the acquisition and preprocessing of data sets. It also includes a tool utils for various tasks: latent space visualization, evaluation, etc.

- **metrics:** Its main functionality is the creation of new custom metrics. From an abstract class on which to inherit and create new metrics.

- **models:** This is the core module of the library. It includes the base architectures on which new ones can be created, several predefined architectures and a list of predefined default encoders and decoders,

- **pipelines:** Pipelines are designed to perform a specific task such as data preprocessing or model training. 

## Installation ⚙️

### With Package manager Pip
The easiest way is to use the pip command so that it's installed together with all its dependencies.

```bash
  pip install rapidae
```

### From source code
You can also download this repository and then create a virtual environment to install the dependencies in.
We recommend this option if you plan to contribute to Conmo.

```bash
git clone https://github.com/NahuelCostaCortez/rapidae
cd rapidae
```

Then you only have to install the requirements in a new Python virtual environment using:

```bash
pip install -r requirements.txt
```

## Documentation 📚

Check out the full documentation for detailed information on installation, usage, examples and recipes: 🔗 [Documentation Link](https://rapidae.readthedocs.io/en/latest/)

All documentation source and configuration files are located inside the docs directory.


## Dealing with issues 🛠️	

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/NahuelCostaCortez/rapidae/issues).


## License ✒️

This project is licensed under the Apache 2.0 license. See LICENSE file for further details.
