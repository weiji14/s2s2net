# S2S2Net

Sentinel-2 Super-Resolution Segmentation Network

# Getting started

## Installation

### Basic

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation.html)
to install both non-python binaries and python libraries.
A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd s2s2net
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate s2s2net

Finally, double-check that the libraries have been installed.

    mamba list

### Advanced

This is for those who want full reproducibility of the virtual environment.

Making an explicit [conda-lock](https://github.com/conda-incubator/conda-lock) file
(only needed if creating a new virtual environment/refreshing an existing one).

    mamba env create --file environment.yml
    mamba list --explicit > environment-linux-64.lock

Creating/Installing a virtual environment from a conda lock file.
See also https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/managenvironments.html#building-identical-conda-environments.

    mamba create --name s2s2net --file environment-linux-64.lock
    mamba install --name s2s2net --file environment-linux-64.lock

## Running jupyter lab

    mamba activate s2s2net
    python -m ipykernel install --user --name s2s2net  # to install virtual env properly
    jupyter kernelspec list --json                     # see if kernel is installed
    jupyter lab &
