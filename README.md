# scHierarchy: Hierarchical logistic regression model for marker gene selection using hierachical cell annotations

[![Stars](https://img.shields.io/github/stars/dissatisfaction-ai/scHierarchy?logo=GitHub&color=yellow)](https://github.com/vitkl/scHierarchy/stargazers)
[![Documentation Status](https://readthedocs.org/projects/scHierarchy/badge/?version=latest)](https://scHierarchy.readthedocs.io/en/stable/?badge=stable)
![Build Status](https://github.com/dissatisfaction-ai/scHierarchy/actions/workflows/test.yml/badge.svg?event=push)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

<img width="708" alt="image" src="https://user-images.githubusercontent.com/22567383/185173791-599778e0-89d4-4b68-823f-3913a66156e9.png">

## Installation

Linux installation
```bash
conda create -y -n schierarchy-env python=3.9
conda activate schierarchy-env
pip install git+https://github.com/dissatisfaction-ai/scHierarchy.git
```

Mac installation
```bash
conda create -y -n schierarchy-env python=3.8
conda activate schierarchy-env
pip install git+https://github.com/pyro-ppl/pyro.git@dev
conda install -y -c anaconda hdf5 pytables netcdf4
pip install git+https://github.com/dissatisfaction-ai/scHierarchy.git
```

## Usage example notebook

https://github.com/dissatisfaction-ai/scHierarchy/blob/main/docs/notebooks/marker_selection_example.ipynb

This will be updated using a publicly available dataset & Colab - however, it is challenging to find published dataset with several annotation levels yet sufficiently small to be used on Colab.
