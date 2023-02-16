import yaml


def load_yaml_config():
    with open("config.yaml", "r", encoding="utf8") as stream:
        yaml_config = yaml.safe_load(stream)
    return yaml_config


yaml_return = load_yaml_config()
print(yaml_return)
