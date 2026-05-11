"""Default dataset name lists for benchmark suite scripts."""

from __future__ import print_function

from utils.exptable2graph_loader import EXPTABLE2GRAPH_SUBDIRS

# Eight benchmarks; disease/telecom keys differ by task (see *_nc vs *_lp).
BENCHMARK8_NC = (
    'cora',
    'Actor',
    'airport',
    'citeseer',
    'cornell',
    'disease_nc',
    'pubmed',
    'telecom_nc',
)
BENCHMARK8_LP = (
    'cora',
    'Actor',
    'airport',
    'citeseer',
    'cornell',
    'disease_lp',
    'pubmed',
    'telecom_lp',
)

STANDARD_BENCHMARK_DATASETS = BENCHMARK8_NC


def default_exptable_dataset_keys():
    return sorted(EXPTABLE2GRAPH_SUBDIRS.keys())


def default_standard_dataset_names(task='nc'):
    """Eight dataset string names for NC or LP."""
    t = (task or 'nc').lower()
    if t == 'lp':
        return list(BENCHMARK8_LP)
    return list(BENCHMARK8_NC)
