import json
import os
import subprocess
import sys
from pathlib import Path


def _run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ''


def query_nvidia_smi(gpu_id=0):
    output = _run_command([
        'nvidia-smi',
        f'--query-gpu=index,name,memory.total',
        '--format=csv,noheader,nounits',
    ])
    if not output:
        return None

    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            rows.append({
                'index': int(parts[0]),
                'name': parts[1],
                'memory_total_mb': int(parts[2]),
            })
        except Exception:
            continue

    for row in rows:
        if row['index'] == gpu_id:
            return row
    return rows[0] if rows else None


def query_torch_cuda(gpu_id=0):
    script = (
        'import json\n'
        'try:\n'
        ' import torch\n'
        ' if not torch.cuda.is_available():\n'
        '  print(json.dumps({"available": False}))\n'
        ' else:\n'
        '  idx = min(' + str(gpu_id) + ', torch.cuda.device_count() - 1)\n'
        '  props = torch.cuda.get_device_properties(idx)\n'
        '  print(json.dumps({"available": True, "index": idx, "name": props.name, "memory_total_mb": int(props.total_memory / 1024 / 1024)}))\n'
        'except Exception:\n'
        ' print(json.dumps({"available": False}))\n'
    )
    output = _run_command([sys.executable, '-c', script])
    if not output:
        return None
    try:
        payload = json.loads(output)
    except Exception:
        return None
    if not payload.get('available'):
        return None
    return {
        'index': int(payload['index']),
        'name': payload['name'],
        'memory_total_mb': int(payload['memory_total_mb']),
    }


def detect_gpu_info(gpu_id=0):
    return query_nvidia_smi(gpu_id) or query_torch_cuda(gpu_id)


def classify_gpu_tier(memory_total_mb):
    if memory_total_mb is None:
        return 'unknown'
    if memory_total_mb <= 12288:
        return 'low'
    if memory_total_mb <= 24576:
        return 'mid'
    return 'high'


def build_gpu_profile(gpu_id=0):
    info = detect_gpu_info(gpu_id)
    memory_total_mb = info['memory_total_mb'] if info else None
    tier = classify_gpu_tier(memory_total_mb)
    return {
        'gpu_id': gpu_id,
        'gpu_name': info['name'] if info else None,
        'memory_total_mb': memory_total_mb,
        'tier': tier,
        'detected': info is not None,
    }


def dataset_scale(dataset=None):
    dataset_lower = (dataset or '').lower()
    if dataset_lower in {'telecom', 'f1_ultimate_hetero_graph'}:
        return 'large'
    if dataset_lower in {
        'actor',
        'cs_phds',
        'carcinogenesis_data',
        'cornell',
        'disease_nc',
        'disease_lp',
        'hepatitis_std_data',
        'hockey_data',
        'pte',
        'toxicology_data',
    }:
        return 'medium'
    return 'small'


def scale_trial_count(base_trials, gpu_profile, dataset=None):
    return int(base_trials)


def apply_hgcn_train_constraints(params, gpu_profile, dataset=None, task=None):
    return dict(params)


def apply_graphmore_train_constraints(params, gpu_profile, dataset=None, task=None):
    return dict(params)


def apply_qgcn_train_constraints(params, gpu_profile, dataset=None, task=None, model=None):
    return dict(params)
