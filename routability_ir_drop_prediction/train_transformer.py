import os
import os.path as osp
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import build_metric, build_roc_prc_metric
import optuna

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
import sys
import numpy as np

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                      end: float,
                      factor: float,
                      weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                         f'cumulative_periods {cumulative_periods}')

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
                            ]


def train(arg_dict, 
        #   FEATURE_SIZE, NUM_HEADS, MLP_SIZE, 
          LR, WEIGHT_DECAY, trial_number):
    save_path = "trail_" + str(trial_number)
    arg_dict['save_path'] = save_path
    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False
    # arg_dict['FEATURE_SIZE'] = FEATURE_SIZE
    # arg_dict['NUM_HEADS'] = NUM_HEADS
    # arg_dict['MLP_SIZE'] = MLP_SIZE
    # arg_dict['NUM_ENCODERS'] = NUM_ENCODERS

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

    # Build lr scheduler
    cosine_lr = CosineRestartLr([LR], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    # save_freq = 1000

    log_output_path = f"{arg_dict['save_path']}/logs"
    if not os.path.exists(log_output_path):
        os.makedirs(log_output_path)
    writer = SummaryWriter(log_dir=log_output_path)

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1

                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        writer.add_scalar('Loss/train', epoch_loss / print_freq, iter_num)
        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        # if iter_num % save_freq == 0:
        #     checkpoint(model, iter_num, arg_dict['save_path'])
        epoch_loss = 0

    writer.close()

    return model


def validate(model, arg_dict):
    model.eval()

    arg_dict['dataset_type'] = 'CongestionDataset'
    arg_dict['ann_file'] = './files/test_N28.csv'
    arg_dict['test_mode'] = True
    arg_dict['eval_metric'] = ['NRMS', 'SSIM']

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    if not arg_dict['cpu']:
        model = model.cuda()

    # Build metrics
    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0 for k in arg_dict['eval_metric']}

    for feature, label, _ in dataset:

        if arg_dict['cpu']:
            input, target = feature, label
        else:
            input, target = feature.cuda(), label.cuda()

        prediction = model(input)
        for metric, metric_func in metrics.items():
            if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    return avg_metrics['SSIM'] 


def objective(trial):
    # Suggest hyperparameters
    # PATCH_SIZE = trial.suggest_int('PATCH_SIZE', 4, 16)
    # FEATURE_SIZE = trial.suggest_int('FEATURE_SIZE', 32, 128)
    # NUM_HEADS = trial.suggest_int('NUM_HEADS', 4, 16)
    # FEATURE_SIZE = trial.suggest_categorical('FEATURE_SIZE', [i for i in range(32, 129) if i % NUM_HEADS == 0])
    # MLP_SIZE = trial.suggest_int('MLP_SIZE', 64, 256)
    # NUM_ENCODERS = trial.suggest_int('NUM_ENCODERS', 3, 12)
    LR = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    WEIGHT_DECAY = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    model = train(arg_dict, 
                # PATCH_SIZE, 
                # FEATURE_SIZE,
                # NUM_HEADS, MLP_SIZE,
                # NUM_ENCODERS,
                LR, WEIGHT_DECAY, trial.number)
    val_loss = validate(model, arg_dict)
    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///db.sqlite3',  # Specify the storage URL here.
        study_name='transformer1'
    )
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
