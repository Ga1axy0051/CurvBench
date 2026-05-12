import importlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / 'datasets'
SMOKE_SAVE_DIR = PROJECT_ROOT / 'tmp_smoke_setup'

REQUIRED_PACKAGES = [
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('networkx', 'networkx'),
    ('matplotlib', 'matplotlib'),
    ('sklearn', 'scikit-learn'),
    ('optuna', 'optuna'),
    ('geoopt', 'geoopt'),
    ('torch', 'torch'),
    ('torch_geometric', 'torch-geometric'),
]


def run(cmd):
    print('>>>', ' '.join(str(x) for x in cmd))
    subprocess.check_call(cmd)


def ensure_import(module_name, package_name):
    try:
        importlib.import_module(module_name)
        print(f'[OK] {module_name}')
    except Exception:
        print(f'[INSTALL] {package_name}')
        run([sys.executable, '-m', 'pip', 'install', package_name])
        importlib.import_module(module_name)
        print(f'[OK] {module_name}')


def print_versions():
    import torch
    import torch_geometric
    import geoopt
    import optuna
    import numpy
    import scipy
    import networkx
    import sklearn

    print('\n=== Versions ===')
    print('python          =', sys.version.split()[0])
    print('torch           =', torch.__version__)
    print('torch.cuda      =', torch.version.cuda)
    print('cuda available  =', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda device     =', torch.cuda.get_device_name(0))
    print('torch_geometric =', torch_geometric.__version__)
    print('geoopt          =', geoopt.__version__)
    print('optuna          =', optuna.__version__)
    print('numpy           =', numpy.__version__)
    print('scipy           =', scipy.__version__)
    print('networkx        =', networkx.__version__)
    print('scikit-learn    =', sklearn.__version__)
    print()


def validate_paths():
    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f'Datasets directory not found: {DATASETS_DIR}')

    required_paths = [
        DATASETS_DIR / 'cornell' / 'processed' / 'data.pt',
        DATASETS_DIR / 'Actor' / 'processed' / 'data.pt',
        DATASETS_DIR / 'telecom' / 'telecom_graph.pt',
        DATASETS_DIR / 'disease_nc' / 'disease_nc.edges.csv',
        DATASETS_DIR / 'disease_lp' / 'disease_lp.edges.csv',
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing required dataset files:\n' + '\n'.join(missing))

    print('[OK] Dataset paths validated')


def smoke_test():
    os.makedirs(SMOKE_SAVE_DIR, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'main.py'),
        '--dataset', 'cornell',
        '--downstream_task', 'NC',
        '--root_path', str(DATASETS_DIR),
        '--gpu', '0',
        '--exp_iters', '1',
        '--epochs_cls', '2',
        '--patience_cls', '1',
        '--min_epoch_cls', '0',
        '--epochs_lp', '2',
        '--patience_lp', '1',
        '--min_epoch_lp', '0',
        '--save_dir', str(SMOKE_SAVE_DIR),
    ]
    run(cmd)
    print('[OK] Smoke test passed')


if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    for module_name, package_name in REQUIRED_PACKAGES:
        ensure_import(module_name, package_name)
    print_versions()
    validate_paths()
    smoke_test()
