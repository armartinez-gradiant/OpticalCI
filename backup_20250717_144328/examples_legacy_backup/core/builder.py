import yaml
import torch
from examples.core.models.mzi_cnn import MZI_CLASS_CNN

def build_model(config):
    model_cfg = config['model']
    model = globals()[model_cfg['type']](**model_cfg.get('params', {}))
    return model