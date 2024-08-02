import yaml
from munch import munchify

def get_config():
    return load_yaml('td/config/data.yml')

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return munchify(yaml.load(f, Loader=yaml.FullLoader))