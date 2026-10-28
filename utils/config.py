import yaml


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = yaml.safe_load(val)
    return cfg
