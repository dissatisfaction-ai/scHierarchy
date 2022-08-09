# scHierarchy: A toolking for cell type hierarchies

[![Stars](https://img.shields.io/github/stars/dissatisfaction-ai/scHierarchy?logo=GitHub&color=yellow)](https://github.com/vitkl/scHierarchy/stargazers)
[![Documentation Status](https://readthedocs.org/projects/scHierarchy/badge/?version=latest)](https://scHierarchy.readthedocs.io/en/stable/?badge=stable)
![Build Status](https://github.com/dissatisfaction-ai/scHierarchy/workflows/scHierarchy/badge.svg)
[![codecov](https://codecov.io/gh/dissatisfaction-ai/scHierarchy/branch/main/graph/badge.svg?token=BGI9Z8R11R)](https://codecov.io/gh/vitkl/scHierarchy)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Installation

```bash
git clone git@github.com:vitkl/scHierarchy.git
cd scHierarchy
```

Linux installation
```bash
export PYTHONNOUSERSITE="aaaaa"
conda create -y -n schierarchy_env python=3.9
conda activate schierarchy_env
pip install git+https://github.com/dissatisfaction-ai/scHierarchy.git#egg=scHierarchy[dev,docs,tutorials]
python -m ipykernel install --user --name=schierarchy_env --display-name='Environment (schierarchy_env)'
```

Mac installation
```bash
export PYTHONNOUSERSITE="aaaaa"
conda create -y -n schierarchy_env python=3.8
conda activate schierarchy_env
pip install git+https://github.com/pyro-ppl/pyro.git@dev
conda install -y -c anaconda hdf5 pytables netcdf4
pip install git+https://github.com/dissatisfaction-ai/scHierarchy.git#egg=scHierarchy[dev,docs,tutorials]
python -m ipykernel install --user --name=schierarchy_env --display-name='Environment (schierarchy_env)'
```
