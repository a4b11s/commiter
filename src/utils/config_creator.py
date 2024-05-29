import hashlib
import json
import os

#TODO: add function to get config from yml file
class ConfigCreator:
    def __init__(self, config_field, default_config, config_file_path = None):
        self.config_field = config_field
        self.config_file_path = config_file_path
        
        self.validation(default_config)
        
        self.default_config = default_config
        self.yml_config = self.get_yml_config()
    
    def validation(self, config):
        for field in config.keys():
            if field not in self.config_field:
                raise ValueError(f'Field {field} not in config_field')
    
    def get_yml_config(self):
        return None
    
    @property
    def config(self):
        return self.yml_config if self.yml_config is not None else self.default_config
    
    @property
    def config_hash(self):
        config_b = json.dumps(self.config, sort_keys=True).encode('utf-8')
        return hashlib.sha256(config_b).hexdigest()