import numpy as np

import wandb


class Logger:
    """
    The purpose of this simple logger is to log intermittently and log average values since the last log
    Supports both training and validation epochs with different step counts and intervals.
    """
    def __init__(self, train_log_interval, val_log_interval=None):
        self.train_log_interval = train_log_interval
        self.val_log_interval = val_log_interval
        self.train_data = None
        self.val_data = None
        
    def wandb_init(self, **kwargs):
        try:
            wandb.init(**kwargs)
        except AssertionError:
            print("Wandb init failed! Running in disabled mode")
            wandb.init(mode='disabled')

    def update(self, info, step, phase='train'):
        """
        Update the logger with new info for the given phase (train or val).
        Logs intermittently based on the phase's log interval.
        """
        info = flatten_dict(info)
        
        if phase == 'train':
            if self.train_data is None:
                self.train_data = {key: [] for key in info}
            data = self.train_data
            log_interval = self.train_log_interval
        elif phase == 'val':
            if self.val_data is None:
                self.val_data = {key: [] for key in info}
            data = self.val_data
            log_interval = self.val_log_interval
        else:
            raise ValueError("Unknown phase: choose 'train' or 'val'")
        
        # Update data for the current phase
        for key in info:
            data[key].append(info[key])
        
        # Log if it's the right step
        if step % log_interval == 0:
            means = {key: np.mean(value) for key, value in data.items()}
            self.log(means, step, phase)
            # Reset the data after logging
            if phase == 'train':
                self.train_data = None
            elif phase == 'val':
                self.val_data = None

    def log(self, info, step, phase='train', flatten=True):
        """
        Logs the averaged information to wandb.
        """
        info[f'{phase}_step'] = step
        if flatten:
            info = flatten_dict(info)
        wandb.log(info)


def flatten_dict(in_dict):
    """
    The purpose of this is to flatten dictionaries because as of writing wandb handling nested dicts is broken :( 
    https://community.wandb.ai/t/the-wandb-log-function-does-not-treat-nested-dict-as-it-describes-in-the-document/3330
    """
    out_dict = {}
    for key, value in in_dict.items():
        if type(value) is dict:
            for inner_key, inner_value in value.items():
                out_dict[f'{key}/{inner_key}'] = inner_value
        else:
            out_dict[key] = value
    return out_dict