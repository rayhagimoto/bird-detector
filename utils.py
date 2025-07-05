import yaml
from types import SimpleNamespace
import json

class ConfigNamespace(SimpleNamespace):
    """
    A SimpleNamespace subclass that also supports dictionary-style access.
    """
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

def load_config(path):
    """
    Loads a yaml config and returns an object that supports both 
    dictionary-style access (config['data']['path']) and 
    attribute-style access (config.data.path)
    """
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dictionaries to ConfigNamespace objects
    config_json = json.dumps(config_dict)
    config = json.loads(config_json, object_hook=lambda d: ConfigNamespace(**d))
    
    return config