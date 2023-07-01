import os
from typing import Optional, TypedDict, List, cast

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader, Subset


def normalize(df: pd.DataFrame, normalizer: Optional[QuantileTransformer] = None,
              random_state: Optional[np.random.RandomState | int] = None, noise: Optional[float] = 1e-3):
    if normalizer is None:
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(len(df.index) // 30, 1000), 10),
            subsample=int(1e9),
            random_state=random_state,
        )

    if noise is not None:
        stds = np.std(df.values, axis=0, keepdims=True)
        noise_std = noise / np.maximum(stds, noise)
        df += noise_std * np.random.default_rng(random_state).standard_normal(df.values.shape)

    normalizer.fit(df)

    return pd.DataFrame(normalizer.transform(df), columns=df.columns), normalizer


class DatasetSplit(TypedDict):
    train: 'TabularDataset'
    validation: 'TabularDataset'
    test: 'TabularDataset'


def preprocess_and_split(y_col: str, df_train: pd.DataFrame, df_test: Optional[pd.DataFrame],
                         random_state: Optional[np.random.RandomState | int] = None, test_size=0.2,
                         device=None) -> DatasetSplit:
    if df_test is None:
        df_train, df_test = train_test_split(df_train, test_size=test_size, random_state=random_state - 1)

    dataset = TabularDataset(df_train, y_col=y_col, random_state=random_state, device=device)
    encoder_y = dataset.encoder_y
    encoder_x_cat = dataset.encoder_x_cat

    df_train, df_validate = train_test_split(df_train, test_size=test_size, random_state=random_state)

    train_dataset = TabularDataset(df_train, y_col=y_col, random_state=random_state, encoder_y=encoder_y,
                                   encoder_x_cat=encoder_x_cat, device=device)
    x_num_normalize = train_dataset.x_num_normalizer

    validation_dataset = TabularDataset(df_validate, y_col=y_col, random_state=random_state, encoder_y=encoder_y,
                                        encoder_x_cat=encoder_x_cat, x_num_normalizer=x_num_normalize, device=device)
    test_dataset = TabularDataset(df_test, y_col=y_col, random_state=random_state, encoder_y=encoder_y,
                                  encoder_x_cat=encoder_x_cat, x_num_normalizer=x_num_normalize, device=device)

    return DatasetSplit(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
    )


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 y_col: str,
                 random_state: Optional[np.random.RandomState | int] = None,
                 encoder_y: Optional[LabelEncoder] = None,
                 encoder_x_cat: Optional[OrdinalEncoder] = None,
                 x_num_normalizer: Optional[QuantileTransformer] = None,
                 device=None):
        super().__init__()
        assert y_col in df, f'Could not find column "{y_col}" in df'

        self.device = device

        df = df.replace({pd.NA: None})

        # create dependent variable and encode categories if not numeric
        y_df: pd.DataFrame = df[[y_col]]
        self.encoder_y: Optional[LabelEncoder] = None
        self.y: pd.Series = y_df[y_col]

        if not np.issubdtype(y_df.dtypes[y_col], np.number):
            self.encoder_y = encoder_y

            if self.encoder_y is None:
                self.encoder_y = LabelEncoder()
                self.encoder_y.fit(y_df[y_col])
            self.y = pd.Series(self.encoder_y.transform(y_df[y_col]), dtype='int64')
        df = df.drop(columns=[y_col])

        # create x_num and handle missing values
        self.x_num = df.select_dtypes(include=[np.number])
        self.x_num_normalizer: QuantileTransformer = x_num_normalizer
        self.x_num, self.x_num_normalizer = normalize(self.x_num, normalizer=x_num_normalizer,
                                                      random_state=random_state)
        self.x_num = self.x_num.astype('float32')

        # encode categorical values
        self.x_cat = df.select_dtypes(exclude=[np.number])
        self.encoder_x_cat = encoder_x_cat

        if encoder_x_cat is None:
            self.encoder_x_cat = OrdinalEncoder(dtype=int)
            self.encoder_x_cat.fit(self.x_cat)
            append_none_class_to_all = False

            if append_none_class_to_all:
                for i, categories in enumerate(self.encoder_x_cat.categories_):
                    if categories[-1] is not None:
                        self.encoder_x_cat.categories_[i] = np.append(categories, [None])

        self.x_cat = pd.DataFrame(self.encoder_x_cat.transform(self.x_cat), columns=self.x_cat.columns, dtype='int64')

    @property
    def categories(self):
        return list(map(lambda cat: len(cat), self.encoder_x_cat.categories_))

    @property
    def d_numeric(self):
        return len(self.x_num.columns)

    @property
    def n_classes(self):
        if self.encoder_y is not None:
            return len(self.encoder_y.classes_)
        elif self.y.dtype == int:
            return int(self.y.max() + 1)

        return 1

    def __getitem__(self, item: int):
        y = torch.as_tensor(self.y.loc[item].item(), device=self.device)
        x_num = torch.as_tensor(self.x_num.loc[item].values, device=self.device)
        x_cat = torch.as_tensor(self.x_cat.loc[item].values, device=self.device)

        return y, x_num, x_cat

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"TabularDataset(y='{self.y.name}', x_num={list(self.x_num.columns.values)}, x_cat={list(self.x_cat.columns.values)})"

    @staticmethod
    def from_adult(dir_path: str, random_state: Optional[int] = None, device=None):
        names = ['age', 'work_class', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                 'salary']

        train_df = pd.read_csv(os.path.join(dir_path, 'adult.data'), delimiter=', ', header=None, engine='python',
                               names=names)
        test_df: pd.DataFrame = pd.read_csv(os.path.join(dir_path, 'adult.test'), delimiter=', ', header=None,
                                            engine='python', names=names, skiprows=[0])

        for df in [train_df, test_df]:
            df.replace({'?': pd.NA}, inplace=True)
            df.dropna(subset=['salary'], inplace=True)
            df['>50K'] = df['salary'].apply(lambda s: s.replace('.', '') == '>50K')
            df.drop(columns=['salary'], inplace=True)

        return preprocess_and_split(y_col='>50K', df_train=train_df, df_test=test_df, random_state=random_state,
                                    device=device)


class TabularMaskedLMDataset(Dataset):
    def __init__(self, dataset: TabularDataset, mask_prob=0.15):
        """
        :param dataset: the TabularDataset to create a MLM dataset for
        """
        super().__init__()

        dataset: TabularDataset = dataset
        self.y, self.x_num, self.x_cat = next(iter(DataLoader(dataset, batch_size=len(dataset))))

        self.x_num_mask = torch.bernoulli(torch.full(self.x_num.size(), mask_prob))
        self.x_cat_mask = torch.bernoulli(torch.full(self.x_cat.size(), mask_prob))

        if self.x_num_mask.sum().item() == 0 or self.x_cat_mask.sum().item() == 0:
            raise ValueError("Did not mask any values in x_num or x_cat")

    def __getitem__(self, item: int):
        return self.x_num[item], self.x_cat[item], self.x_num_mask[item], self.x_cat_mask[item]

    def __len__(self):
        return len(self.y)


def generate_ple_boundaries(train_dataset: TabularDataset | Subset, n_bins=14) -> torch.Tensor:
    """
    Generate target-aware PLE boundaries for a PiecewiseLinearEncoder class using DecisionTreeClassifier from sklearn.

    :param train_dataset: TabularDataset subset or dataset used for training
    :param n_bins: the number of bins to use per feature
    :return: [n_features x (n_bins + 1)] the boundaries for each bin per numerical feature
    """
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    y, x_num, _ = next(iter(loader))
    device = x_num.device
    y = y.cpu()
    x_num_transpose = x_num.transpose(-1, -2).cpu()

    # try to reduce the number of bins by looking at unique values for each numerical feature
    # min_bin_counts = torch.min(torch.tensor([len(x.unique()) for x in x_num_transpose]),
    #                            n_bins * torch.ones(d_numerical)).dtype(torch.int)
    # if min(min_bin_counts) <= 1:
    #     raise ValueError('Some numerical feature contains the same values')

    # construct decision tree for each feature
    boundaries: List[torch.Tensor] = []
    for column in x_num_transpose:
        tree_classifier = DecisionTreeClassifier(max_leaf_nodes=n_bins)
        tree_classifier.fit(column.unsqueeze(dim=-1), y)
        tree = tree_classifier.tree_

        feature_boundaries = [torch.as_tensor(tree.threshold[i], dtype=torch.float32, device=device) for i in
                              range(tree.node_count) if tree.children_left[i] != tree.children_right[i]]
        feature_boundaries.append(column.min().to(device))
        feature_boundaries.append(column.max().to(device))
        boundaries.append(torch.stack(feature_boundaries).sort().values)

    return torch.stack(boundaries)
