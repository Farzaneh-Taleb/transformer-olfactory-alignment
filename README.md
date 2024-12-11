
# Can Transformers Smell Like Humans?

This repository contains the code for reproducing experimnts in the paper "Can Transformers Smell Like Humans?""




## Installation

Installing a combination of libraries used (deepchem + transformers) in this project might be tricky. Follow these steps to smoothly install everything you need;
```
conda create -n Mol python==3.8.10 -y
conda activate Mol
conda config --add channels conda-forge
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 cudatoolkit-dev=11.0 -c pytorch -c conda-forge
nvcc --version
echo $PATH
echo $CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
echo $CUDA_HOME
conda install rdkit==2021.03.2 pandas=1.2.4 scikit-learn=0.24.2 scipy=1.6.3 -y
pip install transformers==4.6.0 pytorch-lightning==1.1.5 pytorch-fast-transformers==0.4.0 datasets==1.6.2 jupyterlab==3.4.0 ipywidgets==7.7.0 bertviz==1.4.0 packaging
gcc --version
conda install -c conda-forge gcc-9
which gcc && gcc --version
mamba install gxx=9
pip install transformers==4.6.0 pytorch-lightning==1.1.5 pytorch-fast-transformers==0.4.0 datasets==1.6.2 jupyterlab==3.4.0 ipywidgets==7.7.0 bertviz==1.4.0 packaging
cd phd/Olfaction/apex/
git checkout 22.04-dev
Already on '22.04-dev'
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0”
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install deepchem

```

## Reproducing RSA experiments for Snitz and Ravia datasets
### Notebook RSA
embeddings extracted from MoLFormer (embeddings) together with labels (y) is saved as a csv file in "glsf_molformer_embeddings_13_Apr17.csv"

embeddings extracted from MoLFormer (embeddings) together with similarity_ratings (y) is saved as a csv file in "ravia_molformer_embeddings_13_Apr17.csv" and "snitz_molformer_embeddings_13_Apr17.csv"
