import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, TypedDict, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import optuna

from models import Model, ModelForMaskedLM, MLP, ResNet, PiecewiseLinearEncoding, LinearNumericalEncoding, \
    PeriodicEncoding, FTTransformer
from utils.data import TabularDataset, DatasetSplit, TabularMaskedLMDataset, generate_ple_boundaries
from utils.metrics import MetricsTracker, Metrics


class EarlyStopError(Exception):
    pass


class EpochAccuracyRow(TypedDict):
    seed: int
    train: float
    validate: float


def pretrain_masked_lm(model: Model, config: Dict, train_dataset: TabularMaskedLMDataset,
                       val_dataset: TabularMaskedLMDataset, device=None, trial: Optional[optuna.Trial] = None) -> \
        Tuple[float, Model]:
    tqdm.write(f'Performing MaskedLM on {model.__class__.__name__}...')

    patience = config['training'].get('patience_pretrain') or config['training'].get('patience')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    mlm_model = ModelForMaskedLM(model).to(device)
    optimizer = optim.AdamW(mlm_model.parameters(), **config['training']['optimizer'])

    best_state_dict: Optional[Dict] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None

    epochs_progress = tqdm(range(config['training']['n_epochs']), unit='epoch', leave=False)

    for epoch in epochs_progress:
        train_progress = tqdm(train_loader, unit='batch', leave=False)
        mlm_model.train()
        train_loss = 0

        for i, batch in enumerate(train_progress):
            optimizer.zero_grad()

            loss = mlm_model(*batch)

            if torch.isnan(loss).item():
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss
            train_progress.set_description(f"Train Loss: {train_loss / (i + 1):.3f}")

        with torch.no_grad():
            mlm_model.eval()
            val_progress = tqdm(val_loader, unit='batch', leave=False)
            val_loss = 0
            val_steps = 0

            for i, batch in enumerate(val_progress):
                loss = mlm_model(*batch)

                val_steps += 1
                val_loss += loss.item()
                val_progress.set_description(f"Val. Loss: {val_loss / val_steps:.3f}")

            if val_steps == 0:
                raise ValueError("Did not complete any validation steps")

            val_loss /= val_steps

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = mlm_model.state_dict()
                best_epoch = epoch
                epochs_progress.set_description(f"Best Val. Loss: {best_val_loss:.3f}")

            if best_epoch is not None and epoch - best_epoch >= patience:
                tqdm.write(f'Patience limit reached at epoch {epoch}')
                break

            if trial is not None:
                trial.report(val_loss, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    tqdm.write(f"Finished Masked LM with best validation loss {best_val_loss} at epoch {best_epoch}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=False)

    return best_val_loss, model


def test_model(config: Dict, model: nn.Module, split: DatasetSplit):
    device = split["train"].device
    test_loader = DataLoader(split['test'], batch_size=config['training']['batch_size'])

    test_progress = tqdm(test_loader, unit='batch', leave=False)
    test_metrics = MetricsTracker(device=device)

    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(test_progress):
            (y, x_num, x_cat) = batch
            batch_outputs: torch.FloatTensor = model(x_num, x_cat)
            test_metrics.append(y, batch_outputs)

            test_progress.set_description(f"Test Acc.: {test_metrics.accuracy:.3f}")

    return test_metrics.report


@dataclass
class TrainLoopResult:
    config: Dict
    train: Metrics
    validate: Metrics
    test: Optional[Metrics] = None
    accuracies_per_epoch: Optional[pd.DataFrame] = None


def train_and_test(model: nn.Module, config: Dict, split: DatasetSplit, device=None,
                   trial: Optional[optuna.Trial] = None, test=True) -> Tuple[TrainLoopResult, nn.Module]:
    training_loader = DataLoader(split['train'], batch_size=config['training']['batch_size'])
    validation_loader = DataLoader(split['validation'], batch_size=config['training']['batch_size'])

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), **config['training']['optimizer'])

    epochs_progress = tqdm(range(config['training']['n_epochs']), unit='epoch', leave=False)
    best_epoch: Optional[int] = None
    best_metrics: Optional[TrainLoopResult] = None
    best_state_dict: Optional[Dict] = None

    epoch = 0
    epoch_accuracies: List[EpochAccuracyRow] = []

    try:
        for epoch in epochs_progress:
            # train model
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            train_metrics = MetricsTracker(device=device)
            model.train()

            train_loss = 0.0
            train_steps = 0

            for i, batch in enumerate(epoch_progress):
                optimizer.zero_grad()
                (y, x_num, x_cat) = batch

                batch_outputs: torch.FloatTensor = model(x_num, x_cat)

                loss: torch.Tensor = criterion(batch_outputs, y.type(torch.float32))
                train_loss += loss.item()
                train_steps += 1

                train_metrics.append(y, batch_outputs)
                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_metrics.accuracy:.3f}")

                loss.backward()
                optimizer.step()

            # Validate model
            val_metrics = MetricsTracker(device=device)
            epoch_progress = tqdm(validation_loader, unit='batch', leave=False)
            model.eval()

            with torch.no_grad():
                for i, batch in enumerate(epoch_progress):
                    (y, x_num, x_cat) = batch
                    batch_outputs: torch.FloatTensor = model(x_num, x_cat)
                    val_metrics.append(y, batch_outputs)

                    epoch_progress.set_description(f"Val. Acc.: {val_metrics.accuracy:.3f}")

            validation_accuracy = val_metrics.report['accuracy']

            if best_metrics is None or validation_accuracy > best_metrics.validate["accuracy"]:
                epochs_progress.set_description(f"Best Val. Acc.: {validation_accuracy:.3f}")
                best_metrics = TrainLoopResult(
                    train=train_metrics.report,
                    validate=val_metrics.report,
                    config=config,
                )
                best_state_dict = model.state_dict()
                best_epoch = epoch

            if best_epoch is not None and epoch - best_epoch >= config['training']['patience']:
                raise EarlyStopError()

            if trial is not None:
                trial.report(validation_accuracy, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            result: EpochAccuracyRow = {
                "seed": config['training']['seed'],
                "train": train_metrics.report['accuracy'],
                "validate": val_metrics.report['accuracy'],
            }
            epoch_accuracies.append(result)
    except EarlyStopError:
        tqdm.write(f"Patience limit reached at epoch {epoch}")

    epoch += 1

    best_metrics.config['training']['n_epochs_completed'] = epoch
    best_metrics.config['training']['best_epoch'] = best_epoch
    best_metrics.accuracies_per_epoch = pd.DataFrame(epoch_accuracies)

    if best_state_dict is None:
        raise ValueError("Did not obtain best_state_dict")

    model.load_state_dict(best_state_dict)

    if test:
        best_metrics.test = test_model(config, model, split)

    return best_metrics, model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    if torch.cuda.is_available():
        cuda_seed = seed + 3

        if not torch.cuda.is_initialized():
            torch.cuda.init()
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",
                        type=str,
                        choices=[Model.__name__, FTTransformer.__name__, MLP.__name__, ResNet.__name__],
                        default=Model.__name__,
                        help="The model class name to train")
    parser.add_argument("--num-encoding",
                        type=str,
                        choices=[PiecewiseLinearEncoding.__name__,
                                 PeriodicEncoding.__name__,
                                 LinearNumericalEncoding.__name__],
                        default=PiecewiseLinearEncoding.__name__,
                        help=f"The name of the class to use to generate initial embeddings for numerical features (only used when model={Model.__name__})")
    parser.add_argument("--mlm",
                        type=bool,
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help=f"Whether the model should be pre-trained using MaskedLM (only used when model={Model.__name__})")
    parser.add_argument("--bilinear",
                        type=bool,
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help=f"Whether we should use Bilinear Attention (only used when model={Model.__name__})")
    parser.add_argument("--mask-prob", type=float, default=0.30, help="Mask probability for MaskedLM")

    args = parser.parse_args()
    model_name: str = args.model
    encoding_name: str = args.num_encoding
    do_mlm: bool = args.mlm
    mlm_mask_prob: float = args.mask_prob
    use_bilinear: bool = args.bilinear

    dataset_name = "adult"

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device_name}')
    device = torch.device(device_name)

    test_results: List[Metrics] = []
    config: Optional[Dict] = None

    result_dir: Optional[str] = None

    for seed in tqdm(range(15)):
        set_seed(seed)

        split = TabularDataset.from_adult(f"data/raw/{dataset_name}", random_state=seed, device=device)

        d_numerical = split["train"].d_numeric
        d_out = 1 if split["train"].n_classes <= 2 else split["train"].n_classes
        categories = split["train"].categories

        set_seed(seed)

        # get optimal hyperparams for model configuration
        if model_name == MLP.__name__:
            config = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': {
                    'in_features_num': d_numerical,
                    'out_features': d_out,
                    'categories': categories,
                    'embedding_dim': 421,
                    'block_out_features': [42, 503, 503, 503, 111],
                },
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {
                        'lr': 0.003479353084932105,
                        'weight_decay': 0.0,
                    },
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 16,
                }
            }
            model = MLP(**config["model"])
        elif model_name == ResNet.__name__:
            config = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': {
                    'in_features_num': d_numerical,
                    'out_features': d_out,
                    'categories': categories,
                    "d": 339,
                    "d_embedding": 348,
                    "d_hidden_factor": 3.825346478547088,
                    "hidden_dropout": 0.2209100377552283,
                    "n_layers": 8,
                    "residual_dropout": 0.3058115601225451,
                },
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {
                        'lr': 0.003479353084932105,
                        'weight_decay': 0.0,
                    },
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 16,
                }
            }
            model = ResNet(**config["model"])
        elif model_name == FTTransformer.__name__:
            config = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': {
                    'categories': categories,
                    'd_numerical': d_numerical,
                    'd_token': 352,
                    'd_out': d_out,
                    'n_layers': 3,
                    'n_heads': 8,
                    'attention_dropout': 0.290284712609104,
                    'ffn_dropout': 0.1605824872092634,
                    'd_ffn_factor': 2.037608078057796,
                },
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {
                        'lr': 2.670080766240071e-05,
                        'weight_decay': 1.878836874957554e-05,
                    },
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 16,
                }
            }
            model = FTTransformer(**config["model"])
        elif model_name == Model.__name__:
            config = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': {
                    'categories': categories,
                    'd_numerical': d_numerical,
                    'd_token': 352,
                    'd_out': d_out,
                    'n_layers': 3,
                    'n_heads': 8,
                    # 'attention_dropout': 0.2744067519636624,
                    # 'ffn_dropout': 0.35759468318620974,
                    'd_ffn_factor': 2.037608078057796,
                },
                'input_num_name': encoding_name,
                'input_num': {},
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {},
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 8,
                    'patience_pretrain': 4,
                    'do_mlm': do_mlm,
                    'mask_prob': mlm_mask_prob,
                }
            }

            if config['input_num_name'] == PiecewiseLinearEncoding.__name__:
                config['model']['attention_dropout'] = 0.2744067519636624
                config['model']['ffn_dropout'] = 0.35759468318620974
                config['training']['optimizer'] = {
                    'lr': 0.0006431172050131992,
                    'weight_decay': 4.311710058685491e-05,
                }
                config['input_num'] = {
                    'n_bins': 14,
                }
                input_num = PiecewiseLinearEncoding(
                    boundaries=generate_ple_boundaries(**config['input_num'], train_dataset=split["train"]),
                    d_token=config['model']['d_token']
                )
            elif config['input_num_name'] == PeriodicEncoding.__name__:
                config['model']['attention_dropout'] = 0.05913721293446661
                config['model']['ffn_dropout'] = 0.3199605106637619
                config['training']['optimizer'] = {
                    'lr': 2.6919058249260695e-05,
                    'weight_decay': 0.0006823493012435792
                }
                input_num = PeriodicEncoding(
                    in_features=config['model']['d_numerical'],
                    d_token=config['model']['d_token'],
                )
            elif config['input_num_name'] == LinearNumericalEncoding.__name__:
                config['model']['attention_dropout'] = 0.28402228054696615
                config['model']['ffn_dropout'] = 0.4627983191463305
                config['training']['optimizer'] = {
                    'lr': 1.633458761106948e-05,
                    'weight_decay': 1.8255254802399014e-06,
                }
                input_num = LinearNumericalEncoding(
                    in_features=config['model']['d_numerical'],
                    d_token=config['model']['d_token'],
                )
            else:
                raise ValueError(f"Unsupported input_num_name {config['input_num_name']}")

            model = Model(**config['model'], input_num=input_num, bilinear=use_bilinear)
            if do_mlm:
                mask_prob = config['training'].get('mask_prob') or 0.15
                train_mlm_dataset = TabularMaskedLMDataset(split["train"], mask_prob=mask_prob)
                val_mlm_dataset = TabularMaskedLMDataset(split["validation"], mask_prob=mask_prob)

                _, model = pretrain_masked_lm(model, config, train_dataset=train_mlm_dataset,
                                              val_dataset=val_mlm_dataset, device=device_name)
        else:
            raise ValueError("Unsupported model_name")

        model_dir = config['model_name']
        if config['model_name'] == Model.__name__:
            model_dir = f"{model_dir}-{config['input_num_name']}"
            if do_mlm:
                model_dir += f"-MLM{mlm_mask_prob * 100:.0f}"
            elif not use_bilinear:
                model_dir += '-NoBA'

        result_dir = os.path.join("results", dataset_name, model_dir)
        result_dir_seed = os.path.join(result_dir, str(config['training']['seed']))
        result_dir_seed_results = os.path.join(result_dir_seed, 'results.json')

        if seed == 0:
            tqdm.write(f"Storing results in {result_dir}")

        if os.path.exists(result_dir_seed):
            with open(os.path.join(result_dir_seed, 'config.json'), 'r') as f:
                config = json.load(f)

            with open(result_dir_seed_results, 'r') as f:
                results: Dict[str, Metrics] = json.load(f)
                test_result = results['test']

            tqdm.write(
                f"Found existing result (Test Acc.: {test_result['accuracy']:.3f}, Epochs: {config['training']['n_epochs_completed']})")
            test_results.append(test_result)
            continue

        tqdm.write(f"Training {model.__class__.__name__}...")
        start = time.process_time()
        result, model = train_and_test(model, config, split, device=device_name)
        duration_sec = time.process_time() - start
        result.config['duration_sec'] = duration_sec

        tqdm.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]: Training finished for seed {seed} with test accuracy {result.test['accuracy']}")

        test_results.append(result.test)

        os.makedirs(result_dir_seed, exist_ok=True)
        with open(os.path.join(result_dir_seed, 'state_dict.pt'), "wb") as f:
            torch.save(model.state_dict(), f)

        with open(os.path.join(result_dir_seed, 'results.json'), 'w') as f:
            json.dump({
                'train': result.train,
                'validate': result.validate,
                'test': result.test,
            }, f, indent=2)

        with open(os.path.join(result_dir_seed, 'config.json'), 'w') as f:
            json.dump(result.config, f, indent=2)

        if result.accuracies_per_epoch is not None:
            result.accuracies_per_epoch.to_csv(os.path.join(result_dir, "accuracies.csv"),
                                               mode='a' if config['training']['seed'] > 0 else 'w',
                                               header=config['training']['seed'] == 0)

    print(f"\nAverage accuracy: {sum(map(lambda item: item['accuracy'], test_results)) / len(test_results)}")
    if result_dir is not None:
        pd.DataFrame(test_results).to_csv(os.path.join(result_dir, "results.csv"))


if __name__ == '__main__':
    main()
