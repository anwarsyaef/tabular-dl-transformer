# Deep Learning on Tabular Data

## Getting Started

First, we setup a conda environment with the required dependencies:

```
conda create -n tabular-dl python=3.10 -y
conda activate tabular-dl
conda install pytorch::pytorch=2.0.0 -c pytorch -y
conda install scikit-learn=1.2.2 pandas=2.0.1 tqdm=4.65.0 optuna=3.2.0 -c conda-forge -y
pip install einops==0.6.1
```

While these dependencies are installing, download the [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
and store the files in `data/raw/adult`. This results in the following directory:

- `data/raw/adult`
    - `adult.data`
    - `adult.names`
    - `adult.test`

After installing the dependencies and the dataset, call one of the following scripts:

- `main_hyper.py`: Run hyperparameter tuning, the type of model and encoding can be changed using command line
  arguments, run `python main_hyper.py --help` for more details.
- `main_train.py`: Train a model for different seeds using a given set of hyperparameters. The type of model, and other
  options can be altered using command line arguments. For more details, run `python main_train.py --help`.
