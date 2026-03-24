import yaml


def apply_config_overrides(config, overrides):
    applied = {}
    for override in overrides or []:
        key_path, value = _parse_override(override)
        _set_nested_value(config, key_path, value)
        applied[".".join(key_path)] = value
    return applied


def _parse_override(override):
    if "=" not in override:
        raise ValueError(
            f"Invalid override {override!r}. Expected the form key=value, "
            "for example loss_weights.lambda_rec=1.0"
        )

    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {override!r}. Key cannot be empty.")

    value = yaml.safe_load(raw_value)
    return key.split("."), value


def _set_nested_value(config, key_path, value):
    cursor = config
    for key in key_path[:-1]:
        if key not in cursor:
            cursor[key] = {}
        elif not isinstance(cursor[key], dict):
            raise TypeError(
                f"Cannot set nested override on {'.'.join(key_path)} because {key!r} is not a mapping."
            )
        cursor = cursor[key]

    cursor[key_path[-1]] = value
