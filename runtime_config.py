import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG_FILE = BASE_DIR / 'runtime_config.json'

DEFAULT_RUNTIME_CONFIG = {
    'model_device': 'auto',
    'reranker_device': 'auto',
    'batch_size_cpu': 16,
    'batch_size_gpu': 64,
    'cloud_timeout_sec': 90,
    'cloud_max_file_mb': 1024,
}

_DEVICE_ALIASES = {
    'auto': 'auto',
    'cpu': 'cpu',
    'gpu': 'gpu',
    'cuda': 'cuda',
    'mps': 'mps',
}


def _safe_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def detect_hardware():
    torch = _safe_import_torch()
    info = {
        'torch_available': torch is not None,
        'cuda_available': False,
        'cuda_name': None,
        'mps_available': False,
        'gpu_available': False,
        'preferred_gpu_device': None,
    }

    if torch is None:
        info['preferred_gpu_device'] = 'cpu'
        return info

    try:
        info['cuda_available'] = bool(torch.cuda.is_available())
    except Exception:
        info['cuda_available'] = False

    if info['cuda_available']:
        try:
            info['cuda_name'] = torch.cuda.get_device_name(0)
        except Exception:
            info['cuda_name'] = 'CUDA GPU'

    try:
        mps_backend = getattr(torch.backends, 'mps', None)
        info['mps_available'] = bool(mps_backend and mps_backend.is_available())
    except Exception:
        info['mps_available'] = False

    info['gpu_available'] = info['cuda_available'] or info['mps_available']
    if info['cuda_available']:
        info['preferred_gpu_device'] = 'cuda'
    elif info['mps_available']:
        info['preferred_gpu_device'] = 'mps'
    else:
        info['preferred_gpu_device'] = 'cpu'

    return info


def _normalize_device(value):
    normalized = (value or 'auto').strip().lower()
    return _DEVICE_ALIASES.get(normalized, 'auto')


def load_runtime_config():
    config = dict(DEFAULT_RUNTIME_CONFIG)
    if RUNTIME_CONFIG_FILE.exists():
        try:
            with open(RUNTIME_CONFIG_FILE, 'r', encoding='utf-8') as file_obj:
                stored = json.load(file_obj)
            if isinstance(stored, dict):
                config.update(stored)
        except Exception:
            pass

    env_model = os.getenv('NOVELLECT_MODEL_DEVICE')
    env_reranker = os.getenv('NOVELLECT_RERANKER_DEVICE')
    env_batch_cpu = os.getenv('NOVELLECT_BATCH_SIZE_CPU')
    env_batch_gpu = os.getenv('NOVELLECT_BATCH_SIZE_GPU')
    env_cloud_timeout = os.getenv('NOVELLECT_CLOUD_TIMEOUT_SEC')
    env_cloud_limit = os.getenv('NOVELLECT_CLOUD_MAX_FILE_MB')

    if env_model:
        config['model_device'] = env_model
    if env_reranker:
        config['reranker_device'] = env_reranker
    if env_batch_cpu:
        try:
            config['batch_size_cpu'] = int(env_batch_cpu)
        except Exception:
            pass
    if env_batch_gpu:
        try:
            config['batch_size_gpu'] = int(env_batch_gpu)
        except Exception:
            pass
    if env_cloud_timeout:
        try:
            config['cloud_timeout_sec'] = int(env_cloud_timeout)
        except Exception:
            pass
    if env_cloud_limit:
        try:
            config['cloud_max_file_mb'] = int(env_cloud_limit)
        except Exception:
            pass

    config['model_device'] = _normalize_device(config.get('model_device', 'auto'))
    config['reranker_device'] = _normalize_device(config.get('reranker_device', 'auto'))
    config['batch_size_cpu'] = max(1, int(config.get('batch_size_cpu', DEFAULT_RUNTIME_CONFIG['batch_size_cpu'])))
    config['batch_size_gpu'] = max(1, int(config.get('batch_size_gpu', DEFAULT_RUNTIME_CONFIG['batch_size_gpu'])))
    config['cloud_timeout_sec'] = max(5, int(config.get('cloud_timeout_sec', DEFAULT_RUNTIME_CONFIG['cloud_timeout_sec'])))
    config['cloud_max_file_mb'] = max(1, int(config.get('cloud_max_file_mb', DEFAULT_RUNTIME_CONFIG['cloud_max_file_mb'])))
    return config


def save_runtime_config(config):
    merged = dict(DEFAULT_RUNTIME_CONFIG)
    merged.update(config or {})
    merged['model_device'] = _normalize_device(merged.get('model_device', 'auto'))
    merged['reranker_device'] = _normalize_device(merged.get('reranker_device', 'auto'))
    merged['batch_size_cpu'] = max(1, int(merged.get('batch_size_cpu', DEFAULT_RUNTIME_CONFIG['batch_size_cpu'])))
    merged['batch_size_gpu'] = max(1, int(merged.get('batch_size_gpu', DEFAULT_RUNTIME_CONFIG['batch_size_gpu'])))
    merged['cloud_timeout_sec'] = max(5, int(merged.get('cloud_timeout_sec', DEFAULT_RUNTIME_CONFIG['cloud_timeout_sec'])))
    merged['cloud_max_file_mb'] = max(1, int(merged.get('cloud_max_file_mb', DEFAULT_RUNTIME_CONFIG['cloud_max_file_mb'])))

    with open(RUNTIME_CONFIG_FILE, 'w', encoding='utf-8') as file_obj:
        json.dump(merged, file_obj, ensure_ascii=False, indent=2)
    return merged


def resolve_device(preference='auto'):
    hardware = detect_hardware()
    preference = _normalize_device(preference)

    if preference == 'auto':
        return hardware['preferred_gpu_device'] or 'cpu'
    if preference == 'gpu':
        return hardware['preferred_gpu_device'] or 'cpu'
    if preference == 'cuda':
        return 'cuda' if hardware['cuda_available'] else 'cpu'
    if preference == 'mps':
        return 'mps' if hardware['mps_available'] else 'cpu'
    return 'cpu'


def get_compute_profile():
    config = load_runtime_config()
    hardware = detect_hardware()
    model_device = resolve_device(config['model_device'])
    reranker_device = resolve_device(config['reranker_device'])
    is_gpu = model_device in {'cuda', 'mps'}
    return {
        'config': config,
        'hardware': hardware,
        'model_device': model_device,
        'reranker_device': reranker_device,
        'model_device_requested': config['model_device'],
        'reranker_device_requested': config['reranker_device'],
        'embedding_batch_size': config['batch_size_gpu'] if is_gpu else config['batch_size_cpu'],
        'cloud_timeout_sec': config['cloud_timeout_sec'],
        'cloud_max_file_mb': config['cloud_max_file_mb'],
    }
