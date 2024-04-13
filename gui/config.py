import logging
import configparser

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        try:
            self.config.read(self.config_file)
        except (FileNotFoundError, configparser.Error) as e:
            logging.error(f"Error loading config: {e}")

    def save_config(self):
        try:
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
        except (FileNotFoundError, configparser.Error) as e:
            logging.error(f"Error saving config: {e}")

    def get_value(self, section, option, default=None):
        return self.config.get(section, option, fallback=default)

    def set_value(self, section, option, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][option] = value