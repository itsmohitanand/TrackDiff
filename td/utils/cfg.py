from abc import abstractmethod, ABC
from munch import munchify
import yaml

class BaseConfig:
    def _load_cfg(self, cfg_path):
        with open(cfg_path, 'r') as f:
            return munchify(yaml.load(f, Loader=yaml.FullLoader))

class UserConfig(BaseConfig):
    def __init__(self, username, cfg_path="config/data.yaml"):
        self.cfg = self._load_cfg(cfg_path)
        self.user = username.lower()
        self.user_cfg = self.cfg[self.user]

    def get_cfg(self):

        return self.user_cfg[self.user]

    