import argparse
import json
import os
import pickle

import optuna
import torch

from main_train import set_seed, pretrain_masked_lm, train_and_test
from models import PiecewiseLinearEncoding, PeriodicEncoding, LinearNumericalEncoding, MLP, Model
from utils.data import TabularDataset, generate_ple_boundaries, TabularMaskedLMDataset


class Objective:
    def __init__(self, seed: int, device: str):
        set_seed(seed)

        self.seed = seed
        self.device = device
        self.dataset_name = "adult"
        self.split = TabularDataset.from_adult("data/raw/adult", random_state=seed)

        self.train_mlm_dataset = TabularMaskedLMDataset(self.split["train"])
        self.val_mlm_dataset = TabularMaskedLMDataset(self.split["validation"])

        set_seed(seed)

    def get_checkpoint_dir(self, study_name: str):
        return os.path.join("checkpoints", self.dataset_name, study_name)

    def __call__(self, trial: optuna.Trial) -> float:
        model_name = trial.study.user_attrs["model_name"]
        dataset_name = self.dataset_name
        d_out = 1 if self.split["train"].n_classes <= 2 else self.split["train"].n_classes
        categories = self.split["train"].categories
        d_numerical = self.split["train"].d_numeric
        seed = self.seed

        if model_name == MLP.__name__:
            n_layers = trial.suggest_int("n_layers", 1, 8)
            d_first = [trial.suggest_int("d_layer_first", 1, 512)]
            d_hidden = [trial.suggest_int("d_layer_hidden", 1, 512)] if n_layers > 2 else []
            d_last = [trial.suggest_int("d_layer_last", 1, 512)] if n_layers > 1 else []

            config = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': {
                    'in_features_num': self.split["train"].d_numeric,
                    'out_features': d_out,
                    'categories': categories,
                    'embedding_dim': trial.suggest_int("embedding_dim", 64, 512),
                    'block_out_features': d_first + d_hidden + d_last,
                },
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {
                        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    },
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 16,
                }
            }
            model = MLP(**config["model"])
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
                    'attention_dropout': trial.suggest_float('attention_dropout', 0., 0.5),
                    'ffn_dropout': trial.suggest_float('ffn_dropout', 0., 0.5),
                    'd_ffn_factor': 2.037608078057796,
                },
                'input_num_name': trial.study.user_attrs['encoding_name'],
                'input_num': {},
                'training': {
                    'optimizer_name': 'AdamW',
                    'optimizer': {
                        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    },
                    'seed': seed,
                    'batch_size': 256,
                    'n_epochs': 100,
                    'patience': 8,
                    'patience_pretrain': 4,
                }
            }
            if config['input_num_name'] == PiecewiseLinearEncoding.__name__:
                config['input_num'] = {
                    'n_bins': 14,
                }
                input_num = PiecewiseLinearEncoding(
                    boundaries=generate_ple_boundaries(**config['input_num'], train_dataset=self.split["train"]),
                    d_token=config['model']['d_token']
                )
            elif config['input_num_name'] == PeriodicEncoding.__name__:
                input_num = PeriodicEncoding(
                    in_features=config['model']['d_numerical'],
                    d_token=config['model']['d_token'],
                )
            elif config['input_num_name'] == LinearNumericalEncoding.__name__:
                input_num = LinearNumericalEncoding(
                    in_features=config['model']['d_numerical'],
                    d_token=config['model']['d_token'],
                )
            else:
                raise ValueError(f"Unsupported input_num_name {config['input_num_name']}")

            model = Model(**config['model'], input_num=input_num)
            _, model = pretrain_masked_lm(model, config, train_dataset=self.train_mlm_dataset,
                                          val_dataset=self.val_mlm_dataset, device=self.device)
        else:
            raise ValueError("Unsupported model_name")

        trial.set_user_attr('config', config)

        result, model = train_and_test(model, config, self.split, device=self.device, test=False)

        trial.set_user_attr('config', result.config)
        trial.set_user_attr('results', {
            'train': result.train,
            'validate': result.validate,
            'test': result.test
        })

        checkpoint_dir = self.get_checkpoint_dir(trial.study.study_name)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{trial.number}.pt"))

        return result.validate["accuracy"]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",
                        type=str,
                        choices=[Model.__name__, MLP.__name__],
                        default=Model.__name__,
                        help="The model class name to train")
    parser.add_argument("--num-encoding",
                        type=str,
                        choices=[PiecewiseLinearEncoding.__name__,
                                 PeriodicEncoding.__name__,
                                 LinearNumericalEncoding.__name__],
                        default=PiecewiseLinearEncoding.__name__,
                        help=f"The name of the class to use to generate initial embeddings for numerical features (only used when model={Model.__name__})")

    args = parser.parse_args()
    model_name: str = args.model
    encoding_name: str = args.num_encoding

    seed = 0

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Setting default device to {device_name}')
    device = torch.device(device_name)
    torch.set_default_device(device)

    study_name = f"{model_name}-{encoding_name}"
    objective = Objective(seed=seed, device=device_name)

    checkpoint_dir = objective.get_checkpoint_dir(study_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    study_path = os.path.join(checkpoint_dir, "study.pickle")

    # resume study if exists
    if os.path.isfile(study_path):
        with open(study_path, 'rb') as f:
            print(f"Resuming study from {study_path}")
            study = pickle.load(f)
    else:
        study = optuna.create_study(study_name=study_name,
                                    direction=optuna.study.StudyDirection.MAXIMIZE,
                                    sampler=optuna.samplers.TPESampler(seed=objective.seed))

    study.set_user_attr("model_name", model_name)
    study.set_user_attr("encoding_name", encoding_name)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        best_trial = study.best_trial
        config = best_trial.user_attrs['config']
        os.makedirs(checkpoint_dir, exist_ok=True)

        for t in study.trials:
            state_dict_path = os.path.join(checkpoint_dir, f"{t.number}.pt")

            if t.number != best_trial.number and os.path.exists(state_dict_path):
                os.remove(state_dict_path)

        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(checkpoint_dir, 'results.json'), 'w') as f:
            json.dump(best_trial.user_attrs['results'], f, indent=2)

    study.optimize(objective,
                   n_trials=100 - len(study.trials),
                   callbacks=[callback],
                   show_progress_bar=True)


if __name__ == '__main__':
    main()
