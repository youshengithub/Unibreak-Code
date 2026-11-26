import yaml    
def get_attack_configs(yaml_path="configs/attack_conf.yaml"):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
