import ast

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _merge_missing(config, checkpoint_config, exclude_keys=[]):
    """
    Merges missing keys from the current config into the checkpoint config.

    Args:
    - config (dict): The dictionary with additional configurations.
    - checkpoint_config (dict): The base dictionary to merge into.

    Returns:
    - dict: The merged dictionary.
    """
    for key, value in config.items():
        if key in exclude_keys:
            print(f"Excluding key '{key}' from merging.")
            continue
        if key not in checkpoint_config:
            checkpoint_config[key] = value
            print(f"Adding missing key '{key}' with value '{value}' from current config to checkpoint config.")
        elif isinstance(value, dict) and isinstance(checkpoint_config.get(key), dict):
            _merge_missing(value, checkpoint_config[key])

    return checkpoint_config


def _merge_overrides(config, overrides):
    """
    Merges the override dictionary into the config dictionary, accounting for nesting.

    Args:
    - config (dict): The original config dictionary.
    - overrides (dict): The override dictionary.

    Returns:
    - dict: The updated config dictionary with overrides applied.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            _merge_overrides(config[key], value)
        else:
            config[key] = value
            print(f"Overriding key '{key}' with value '{value}'.")

    return config


def _convert_type(value):
    """
    Override string (yaml) to python type conversion.
    """
    if value=='true':
        return True
    elif value=='false':
        return False
    elif value=='null':
        return None
    elif '[' in value and ']' in value:
        list_elements = value.replace('[','').replace(']','').split(',')
        return [_convert_type(elem) for elem in list_elements]
    else:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        

def _set_nested_value(d, keys, value):
    """
    Sets a value in a nested dictionary given a list of keys.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value



def _parse_overrides(overrides):
    """
    Parses a list of override strings into a dictionary with proper type conversion.

    Args:
    - overrides (list of str): List of override strings in the format "key=value".

    Returns:
    - dict: A dictionary with the parsed and converted values.
    """
    override_dict = {}
    for override in overrides:
        key, value = override.split('=', 1)
        keys = key.split('.')
        parsed_value = _convert_type(value)
        _set_nested_value(override_dict, keys, parsed_value)
    return override_dict


def merge_checkpoint_config(cfg, ckpt_cfg=None, overrides=True, exclude_keys=[]):
    """
    Merges the config from the checkpoint with the current config.
    If no `ckpt_cfg` arg is provided, config is loaded from `cfg.checkpoint`.

    Rules:
    - The config from the checkpoint is used as the base.
    - Any missing keys are added from the current config.
    - Any overrides specified in command line overwrite the checkpoint config.
    """
    # Load configs as dicts
    cfg_dict = OmegaConf.to_container(cfg)
    if ckpt_cfg is None:
        checkpoint = torch.load(cfg.checkpoint)
        ckpt_cfg = OmegaConf.create(checkpoint['config'])
    ckpt_cfg_dict = OmegaConf.to_container(ckpt_cfg)
    print("================== CONFIG FROM CHECKPOINT ==================")
    print(OmegaConf.to_yaml(ckpt_cfg))

    # Merge the two configs
    print("Merging checkpoint config with current config...")
    merged_cfg = _merge_missing(cfg_dict, ckpt_cfg_dict, exclude_keys)

    # Apply overrides
    if overrides:
        overrides_list = HydraConfig.get().overrides.task
        overrides_dict = _parse_overrides(overrides_list)
        print("================== OVERRIDES ==================")
        print(OmegaConf.to_yaml(overrides_dict))

        
        print("Applying overrides...")
        merged_cfg = _merge_overrides(merged_cfg, overrides_dict)

    return OmegaConf.create(merged_cfg)


def copy_keys(source_cfg, dest_cfg, keys):
    """
    Copies values from source config to destination config for specified keys.
    
    Keys are specified as key paths separated by '.'. Cfgs are OmegaConf objects.
    For example, to copy the value of key 'model.vqvae' from source to dest:
    copy_keys(source_cfg, dest_cfg, ['model.vqvae'])
    """
    for key in keys:
        value = OmegaConf.select(source_cfg, key)
        try:
            OmegaConf.update(dest_cfg, key, value)
            print(f"Updating key '{key}' with value '{value}'.")
        except KeyError:
            print(f"Skipping key {key} from checkpoint config.")
        
    return dest_cfg

def get_device(prefer_device: str = None):
    """
    Utility function that can be used from anywhere in the codebase to
    grab the current torch device.
    Args:
        prefer_device: Allows the user to specify a torch device they
        want to use.

    Returns: The preferred torch device if requested, otherwise cuda, or mps, or cpu,
    in that order of preference.

    """
    if prefer_device:
        return torch.device(prefer_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")