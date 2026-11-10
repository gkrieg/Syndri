"""Repository-relative configuration for setup and experiment parameter files."""
from pathlib import Path
import json
from . import ROOT

_SETTINGS = None


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load(path):
    if path.suffix == '.json':
        result = json.loads(path.read_text())
    else:
        import yaml
        result = yaml.safe_load(path.read_text())
    if not isinstance(result, dict):
        raise ValueError('The parameter file must contain a mapping')
    return result


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def configure(path=None):
    global _SETTINGS
    defaults = ROOT / 'configs/reproduction.yaml'
    result = _load(defaults)
    if path is not None and resolve(path) != defaults:
        _merge(result, _load(resolve(path)))
    if not isinstance(result.get('assets'), dict) or not isinstance(result.get('configuration'), dict):
        raise ValueError('Parameter sections assets and configuration must be mappings')
    _SETTINGS = result
    return result


def settings():
    return _SETTINGS if _SETTINGS is not None else configure()


def configuration_path(name, dataset=None):
    value = settings()['configuration'][name]
    return resolve(value[dataset] if dataset is not None else value)
