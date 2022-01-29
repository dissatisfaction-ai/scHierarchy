# scHierarchy: A toolking for cell type hierarchies

[![Stars](https://img.shields.io/github/stars/vitkl/scHierarchy?logo=GitHub&color=yellow)](https://github.com/vitkl/scHierarchy/stargazers)
[![Documentation Status](https://readthedocs.org/projects/scHierarchy/badge/?version=latest)](https://scHierarchy.readthedocs.io/en/stable/?badge=stable)
![Build Status](https://github.com/vitkl/scHierarchy/workflows/scHierarchy/badge.svg)
[![codecov](https://codecov.io/gh/vitkl/scHierarchy/branch/main/graph/badge.svg?token=BGI9Z8R11R)](https://codecov.io/gh/vitkl/scHierarchy)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Installation

```bash
git clone git@github.com:vitkl/scHierarchy.git
cd scHierarchy
```

Linux installation
```bash
export PYTHONNOUSERSITE="aaaaa"
conda create -y -n test_schierarchy python=3.9
conda activate test_schierarchy
pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials]
pip uninstall -y cell2location
pip install git+https://github.com/BayraktarLab/cell2location.git@cell2location_hierachical_guide
pip install .[dev,docs,tutorials]
conda activate test_schierarchy
python -m ipykernel install --user --name=test_schierarchy --display-name='Environment (test_schierarchy)'
```

Mac installation
```bash
export PYTHONNOUSERSITE="aaaaa"
conda create -y -n test_schierarchy python=3.8
conda activate test_schierarchy
pip install git+https://github.com/pyro-ppl/pyro.git@dev
conda install -y -c anaconda hdf5 pytables
pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials]
pip uninstall -y cell2location
pip install git+https://github.com/BayraktarLab/cell2location.git@cell2location_hierachical_guide
pip install .[dev,docs,tutorials]
conda activate test_schierarchy
python -m ipykernel install --user --name=test_schierarchy --display-name='Environment (test_schierarchy)'
```
