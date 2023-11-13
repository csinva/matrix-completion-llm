import importlib

def get_configs(config_name):
    dsets = importlib.import_module(f'config.{config_name}.datasets')
    ests = importlib.import_module(f'config.{config_name}.models')
    return dsets.DATALOADERS, ests.ESTIMATORS