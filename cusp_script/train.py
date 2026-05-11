import argparse
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Airports
from torch_geometric.utils import coalesce, negative_sampling, train_test_split_edges, to_undirected
from models.cusp_model import CUSPModel
from layers.cusp_laplacian import CuspLaplacian
from utils.local_airport_dataset import LocalAirportDataset
from utils.disease_dataset import DiseaseDataset
from utils.telecom_dataset import TelecomDataset
from utils.cs_phds_dataset import CsPhdsDataset
from utils.table2graph_dataset import (
    TABLE2GRAPH_FOLDER,
    Table2GraphDataset,
    validate_table2graph_data,
    table2graph_feature_summary,
)
from utils.nc_metrics import nc_split_metrics
import networkx as nx
import numpy as np
import random
import geoopt
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
# import wandb

TABLE2GRAPH_NAMES = tuple(TABLE2GRAPH_FOLDER.keys())


def _geoopt_stereographic_trainable_param_ids(model):
    """
    Parameter ids for geoopt Stereographic-family manifolds (PoincareBall, Sphere, etc.).
    Newer geoopt stores curvature as isp_c / isp_k (not bare .c/.k), so name-suffix matching fails.
    """
    ids = set()
    try:
        from geoopt.manifolds.stereographic.manifold import Stereographic
    except ImportError:
        try:
            from geoopt.manifolds.stereographic import Stereographic
        except ImportError:
            return ids
    for mod in model.modules():
        if isinstance(mod, Stereographic):
            for p in mod.parameters():
                if p.requires_grad:
                    ids.add(id(p))
    if not ids:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            last = name.rsplit(".", 1)[-1]
            if last in ("isp_c", "isp_k", "c", "k", "_c", "_k"):
                ids.add(id(p))
    return ids


_DATASET_CHOICES = [
    "Cora", "Citeseer", "PubMed", "Chameleon", "Actor", "Squirrel", "Texas", "Cornell",
    "AirportUSA", "AirportBrazil", "AirportEurope", "AirportLocal", "Disease", "Telecom", "cs_phds",
] + list(TABLE2GRAPH_NAMES)


def _patch_pyg_dataset_no_download():
    """If raw files are missing, block PyG auto-download (offline-friendly)."""
    import torch_geometric.data.dataset as ds

    if getattr(ds.Dataset, "_cusp_no_download_patch", False):
        return
    _orig = ds.Dataset._download

    def _download(self):
        if ds.files_exist(self.raw_paths):
            return
        names = self.raw_file_names
        if callable(names):
            names = names()
        raise RuntimeError(
            "Offline mode: raw files missing or mismatch; download disabled.\n"
            "  raw_dir: %s\n"
            "  expected: %s\n"
            "Place files there or pass --allow_download."
            % (self.raw_dir, names)
        )

    ds.Dataset._download = _download
    ds.Dataset._cusp_no_download_patch = True


def _argv_has_flag(argv, flag):
    """True if argv contains ``--flag`` or ``--flag=...`` (keep CLI overrides when merging config)."""
    pfx = flag + "="
    for a in argv:
        if a == flag or a.startswith(pfx):
            return True
    return False


def apply_config_json(args, path):
    """Merge config.json ``cusp_training`` (and related) into ``args`` for the chosen dataset."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    exp = cfg.get("experiment") or {}
    if getattr(args, "seeds", None) is None and exp.get("default_seeds"):
        args.seeds = list(exp["default_seeds"])
    ds_cfg = (cfg.get("benchmark_datasets") or {}).get(args.dataset)
    if ds_cfg is None:
        ds_cfg = (cfg.get("table2graph_datasets") or {}).get(args.dataset)
    if isinstance(ds_cfg, dict):
        cusp = ds_cfg.get("cusp_training") or ds_cfg.get("training")
        if isinstance(cusp, dict):
            for k, v in cusp.items():
                if hasattr(args, k):
                    setattr(args, k, v)


def reapply_cli_train_span(args, argv, snap):
    """Restore epochs/patience/min_delta from CLI if those flags were passed."""
    e, p, m = snap
    if _argv_has_flag(argv, "--epochs"):
        args.epochs = e
    if _argv_has_flag(argv, "--patience"):
        args.patience = p
    if _argv_has_flag(argv, "--min_delta"):
        args.min_delta = m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def export_hparams_dict(args, data_root):
    """Hyperparameters and run metadata for JSON export / reproducibility."""
    h = {
        "config_json": getattr(args, "config", None),
        "data_root": os.path.normpath(data_root),
        "dataset": args.dataset,
        "task": args.task,
        "model": args.model,
        "manifold_config": args.manifold_config,
        "K_GPR": args.K,
        "num_propagation_orders": args.K + 1,
        "alpha_GPR": args.alpha,
        "Init_GPR_weights": args.Init,
        "Gamma_GPR": args.Gamma,
        "d_f_curvature": args.d_f,
        "num_frequencies": args.num_frequencies,
        "dropout": args.dropout,
        "dprate": args.dprate,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "lr_scheduler": {"type": "StepLR", "step_size": 50, "gamma": 0.1},
        "loss": "cross_entropy" if args.task == "node_classification" else "link_prediction (BCE with logits)",
        "activation_in_cusp_gnn": "ReLU",
        "output_layer": "Linear + log_softmax (node classification)",
        "use_cusp_laplacian": args.use_cusp_laplacian,
        "use_curvature_encoding": args.use_curvature_encoding,
        "use_cusp_pooling": args.use_cusp_pooling,
        "euclidean_variant": args.euclidean_variant,
        "ricci_alpha": args.ricci_alpha,
        "allow_download": args.allow_download,
        "device": args.device,
        "notes": "CUSP uses GPR filter bank (K) and manifold_config; not a deep stack of message-passing layers.",
    }
    if getattr(args, "dataset", None) == "cs_phds":
        croot = getattr(args, "cs_phds_root", None) or os.path.join(data_root, "cs_phds")
        nc_sub = getattr(args, "cs_phds_nc_subdir", None) or "cs_phds_nc_ready"
        lp_sub = getattr(args, "cs_phds_lp_subdir", None) or "cs_phds_lp_ready"
        h["cs_phds"] = {
            "root": os.path.normpath(croot),
            "node_classification_dir": os.path.normpath(os.path.join(croot, nc_sub)),
            "link_prediction_dir": os.path.normpath(os.path.join(croot, lp_sub)),
        }
    return h


def load_dataset_for_args(args, data_root, t2g_root):
    """Load only ``args.dataset`` (avoid touching unrelated benchmarks)."""
    dr = data_root
    if args.dataset == "AirportLocal":
        ar = args.airport_root or os.path.join(dr, "Airport")
        return LocalAirportDataset(root=ar, gdp_bins=args.gdp_bins)
    if args.dataset == "Disease":
        disease_r = args.disease_root or os.path.join(dr, "disease")
        return DiseaseDataset(root=disease_r)
    if args.dataset == "Telecom":
        tr = args.telecom_root or os.path.join(dr, "telecom")
        return TelecomDataset(root=tr)
    if args.dataset == "cs_phds":
        cpr = args.cs_phds_root or os.path.join(dr, "cs_phds")
        return CsPhdsDataset(
            root=cpr,
            task=args.task,
            nc_subdir=args.cs_phds_nc_subdir or "cs_phds_nc_ready",
            lp_subdir=args.cs_phds_lp_subdir or "cs_phds_lp_ready",
            verbose=True,
        )
    if args.dataset in TABLE2GRAPH_NAMES:
        return Table2GraphDataset(name=args.dataset, root=t2g_root)

    if args.dataset == "Cora":
        return Planetoid(root=os.path.join(dr, "Cora"), name="Cora", transform=T.ToUndirected())
    if args.dataset == "Citeseer":
        return Planetoid(root=os.path.join(dr, "Citeseer"), name="Citeseer", transform=T.ToUndirected())
    if args.dataset == "PubMed":
        return Planetoid(root=os.path.join(dr, "PubMed"), name="PubMed", transform=T.ToUndirected())
    if args.dataset == "Chameleon":
        return WikipediaNetwork(root=os.path.join(dr, "WikipediaNetwork"), name="chameleon", transform=T.ToUndirected())
    if args.dataset == "Actor":
        return Actor(root=os.path.join(dr, "Actor"), transform=T.ToUndirected())
    if args.dataset == "Squirrel":
        return WikipediaNetwork(root=os.path.join(dr, "WikipediaNetwork"), name="squirrel", transform=T.ToUndirected())
    if args.dataset == "Texas":
        return WebKB(root=dr, name="Texas", transform=T.ToUndirected())
    if args.dataset == "Cornell":
        return WebKB(root=dr, name="Cornell", transform=T.ToUndirected())
    if args.dataset == "AirportUSA":
        return Airports(root=os.path.join(dr, "airport"), name="usa", transform=T.ToUndirected())
    if args.dataset == "AirportBrazil":
        return Airports(root=os.path.join(dr, "airport"), name="brazil", transform=T.ToUndirected())
    if args.dataset == "AirportEurope":
        return Airports(root=os.path.join(dr, "airport"), name="europe", transform=T.ToUndirected())

    raise ValueError("Unsupported dataset: %s" % args.dataset)


def stratified_nc_split(data, seed, train_ratio=0.6):
    """60/20/20 stratified masks when dataset has no train_mask (e.g. PyG Airports)."""
    num_nodes = data.num_nodes
    y = data.y.view(-1).cpu().numpy()
    # Split only labeled nodes (y >= 0); y == -1 is ignored
    labeled = y >= 0
    if not labeled.any():
        raise ValueError("stratified_nc_split: no labeled nodes (need y >= 0).")
    indices = np.flatnonzero(labeled)
    y_lab = y[indices]
    val_test_ratio = 1.0 - train_ratio
    try:
        idx_train, idx_tmp = train_test_split(
            indices, test_size=val_test_ratio, stratify=y_lab, random_state=seed
        )
        y_tmp = y[idx_tmp]
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=0.5, stratify=y_tmp, random_state=seed + 1
        )
    except ValueError:
        idx_train, idx_tmp = train_test_split(
            indices, test_size=val_test_ratio, random_state=seed
        )
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=0.5, random_state=seed + 1
        )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.as_tensor(idx_train, dtype=torch.long)] = True
    val_mask[torch.as_tensor(idx_val, dtype=torch.long)] = True
    test_mask[torch.as_tensor(idx_test, dtype=torch.long)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def main():
    parser = argparse.ArgumentParser(description="CUSP Model Training with Node Classification and Link Prediction")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of experiment runs') 
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dataset', type=str, default='Cora', choices=_DATASET_CHOICES, help='Benchmark or Table2Graph name; see config.json and utils/*_dataset.py')
    parser.add_argument('--model', type=str, default='cusp', choices=['cusp'], help='Model to use (CUSP)')
    parser.add_argument('--manifold_config', type=str, default='H16H16S16E16', help='Product manifold signature in the form of a string.')
    parser.add_argument('--K', type=int, default=10, help='Number of filters in the filterbank.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for GPR propagation')
    parser.add_argument('--Init', type=str, default='PPR', choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS'], help='Initialization method for GPR weights')
    parser.add_argument('--Gamma', type=float, default=None, help='Gamma parameter for GPR weights')
    parser.add_argument('--d_f', type=int, default=64, help='Dimensionality of curvature embeddings.')
    parser.add_argument('--num_frequencies', type=int, default=16, help='Number of frequencies for curvature encoding')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dprate', type=float, default=0.5, help='Dropout rate for propagation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping: stop after this many epochs without improvement on the monitored metric; 0 = disabled (always run all epochs).')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum improvement to reset patience (NC: val F1; LP: test AUC).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'radam'],
        help='adam: for CUSP with learnable curvature, train.py uses geoopt RiemannianAdam (stable); radam is the same. '
        'Use --euclidean_variant if you need Euclidean torch.optim.Adam on the full CUSP stack.',
    )
    parser.add_argument('--ricci_alpha', type=float, default=0.5, help='Alpha parameter for Ollivier-Ricci curvature')
    parser.add_argument('--task', type=str, default='node_classification', choices=['node_classification', 'link_prediction'], help='Task to perform')
    parser.add_argument('--use_cusp_laplacian', action='store_true', help='Use Cusp Laplacian (default). If not set, uses standard graph Laplacian.')
    parser.add_argument('--use_curvature_encoding', action='store_true', help='Use curvature-based positional encoding in Cusp Pooling.')
    parser.add_argument('--use_cusp_pooling', action='store_true', help='Use Cusp Pooling with hierarchical attention. If not set, uses simple embedding concatenation.')
    parser.add_argument('--euclidean_variant', action='store_true', help='Use Euclidean variant of the model (all manifolds are Euclidean).')
    parser.add_argument('--wandb_project', type=str, default='CUSP_GNN', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity (team/user)')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--ppnp', type=str, default='GPR_prop', choices=['PPNP', 'GPR_prop'], help='Propagation method')
    parser.add_argument('--airport_root', type=str, default=None, help='For AirportLocal only: folder with airport_edgelist.txt (default: data/Airport)')
    parser.add_argument('--gdp_bins', type=int, default=10, help='For AirportLocal + airport_alldata.p: quantile bins for GDP as class labels (NC)')
    parser.add_argument('--disease_root', type=str, default=None, help='For Disease only: folder with disease_nc.edges.csv (default: data/disease)')
    parser.add_argument('--telecom_root', type=str, default=None, help='For Telecom only: folder with edges csv (default: data/telecom)')
    parser.add_argument(
        '--cs_phds_root',
        type=str,
        default=None,
        help='cs_phds parent dir (default <data_root>/cs_phds); contains cs_phds_nc_ready and cs_phds_lp_ready',
    )
    parser.add_argument(
        '--cs_phds_nc_subdir',
        type=str,
        default=None,
        help='cs_phds NC subfolder name (default cs_phds_nc_ready)',
    )
    parser.add_argument(
        '--cs_phds_lp_subdir',
        type=str,
        default=None,
        help='cs_phds LP subfolder name (default cs_phds_lp_ready)',
    )
    parser.add_argument('--data_root', type=str, default=None, help='Data root (default ./data or env CUSP_DATA_ROOT)')
    parser.add_argument('--table2graph_root', type=str, default=None, help='Table2Graph root (default <data_root>/exptable2graph)')
    parser.add_argument('--seeds', type=int, nargs='*', default=None, help='Seeds for multi-run; if omitted use --seed and --num_runs')
    parser.add_argument('--config', type=str, default=None, help='Optional config.json path (merges cusp_training per dataset)')
    parser.add_argument('--export_dir', type=str, default=None, help='If set, write per-seed JSON and summary.json here')
    parser.add_argument(
        '--allow_download',
        action='store_true',
        help='Allow PyG to download missing raw files (default: offline, local data only)',
    )
    args = parser.parse_args()
    _train_snap = (args.epochs, args.patience, args.min_delta)

    data_root = args.data_root or os.environ.get("CUSP_DATA_ROOT") or "data"
    data_root = os.path.normpath(data_root)

    if not args.allow_download:
        _patch_pyg_dataset_no_download()

    if args.config:
        apply_config_json(args, args.config)
        reapply_cli_train_span(args, sys.argv, _train_snap)

    t2g_root = args.table2graph_root or os.path.join(data_root, "exptable2graph")
    print("[data] root=%s | offline=%s" % (os.path.normpath(data_root), (not args.allow_download)))

    # Initialize WandB
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Collect metrics over runs
    metrics = []

    if args.seeds is not None and len(args.seeds) > 0:
        run_seeds = list(args.seeds)
    else:
        run_seeds = [args.seed + r for r in range(args.num_runs)]

    for run_idx, current_seed in enumerate(run_seeds):
        print(f"\nRun {run_idx + 1}/{len(run_seeds)} (seed={current_seed})")
        set_seed(current_seed)

        dataset = load_dataset_for_args(args, data_root, t2g_root)
        data = dataset[0]
        if args.dataset in ['Chameleon', 'Actor', 'Squirrel', 'Texas', 'Cornell']:
            #Because there are multiple masks present in these datasets, and we use just one
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
        elif args.task == 'node_classification' and 'train_mask' not in data:
            # No train_mask: stratified 60/20/20 (e.g. Airports, some custom sets)
            data = stratified_nc_split(data, seed=current_seed)
            tag = "Table2Graph" if args.dataset in TABLE2GRAPH_NAMES else "stratified"
            print(f"NC masks: stratified 60/20/20 (seed={current_seed}) [{tag}]")
        if args.dataset in TABLE2GRAPH_NAMES:
            validate_table2graph_data(data, dataset.num_classes, args.dataset)
            print(
                "[Table2Graph %s] %s | N=%d E=%d C=%d"
                % (
                    args.dataset,
                    table2graph_feature_summary(data),
                    data.num_nodes,
                    data.edge_index.size(1),
                    dataset.num_classes,
                )
            )
            # Scale huge features for stereographic stability (global scale only)
            xm = data.x.abs().max().clamp_min(1e-12)
            if float(xm) > 100.0:
                data.x = data.x / xm
                print("[data] Table2Graph: scaled x /= max|x|=%.4f (for stability)" % float(xm))
        elif args.dataset == "cs_phds":
            xm = data.x.abs().max().clamp_min(1e-12)
            if float(xm) > 100.0:
                data.x = data.x / xm
                print("[data] cs_phds: scaled x /= max|x|=%.4f (for stability)" % float(xm))
        num_nodes = data.x.shape[0]

        # Set model input and output dimensions
        input_dim = data.num_features
        output_dim = dataset.num_classes

        # Convert edge_index to NetworkX graph
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        edge_list = edge_index.t().tolist()  # Shape: (E, 2)

        G = nx.Graph()
        G.add_edges_from(edge_list)


        if args.use_cusp_laplacian:
            cusp_laplacian = CuspLaplacian(nx_graph=G, num_nodes = num_nodes, alpha=args.ricci_alpha)
            data.edge_weight = cusp_laplacian.get_ricci_edge_weights(data.edge_index)
            data.kappa = cusp_laplacian.get_curvature_values()  # Curvature values for nodes (N,)
        else:
            # Assign edge weights as ones to recover standard graph Laplacian
            num_edges = data.edge_index.size(1)
            data.edge_weight = torch.ones(num_edges, dtype=torch.float, device=data.edge_index.device)
            data.kappa = torch.zeros(data.num_nodes, dtype=torch.float, device=data.edge_index.device)  # All curvatures are 0 in Euclidean space

        # If task is link prediction, split edges
        if args.task == 'link_prediction':
            # Preserve node features before splitting edges
            x = data.x.clone()

            if getattr(data, "lp_presplit", False):
                # Pre-split LP edges (e.g. cs_phds splits.pt); skip random link split
                data.x = x
                n = data.num_nodes
                for name in ("train_pos_edge_index", "val_pos_edge_index", "test_pos_edge_index"):
                    if hasattr(data, name):
                        ei = getattr(data, name)
                        ei = to_undirected(ei.long().contiguous())
                        co = coalesce(ei, num_nodes=n)
                        setattr(data, name, co[0] if isinstance(co, tuple) else co)
                for name in ("val_neg_edge_index", "test_neg_edge_index"):
                    if hasattr(data, name):
                        ei = getattr(data, name).long().contiguous()
                        co = coalesce(ei, num_nodes=n)
                        setattr(data, name, co[0] if isinstance(co, tuple) else co)
            else:
                # Ensure the graph is undirected
                data.edge_index = to_undirected(data.edge_index)

                # Split edges into train/val/test sets
                data = train_test_split_edges(data)

                # Restore node features
                data.x = x

        # Define model based on the selected argument
        if args.model == 'cusp':
            model = CUSPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                manifold_config_str=args.manifold_config,
                K=args.K,
                alpha=args.alpha,
                Init=args.Init,
                Gamma=args.Gamma,
                d_f=args.d_f,
                num_frequencies=args.num_frequencies,
                dropout=args.dropout,
                dprate=args.dprate,
                use_curvature_encoding=args.use_curvature_encoding,
                use_cusp_pooling=args.use_cusp_pooling,
                euclidean_variant=args.euclidean_variant,
                use_cusp_laplacian=args.use_cusp_laplacian
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        model = model.to(device)
        data = data.to(device)

        # CUSP uses learnable PoincareBall/Sphere curvatures; Euclidean Adam updates them without
        # manifold constraints and often yields NaN after one step. Use RiemannianAdam for cusp
        # unless the model is the pure Euclidean variant (no learnable curvature).
        use_riemannian = (args.optimizer == 'radam') or (
            args.optimizer == 'adam' and args.model == 'cusp' and not args.euclidean_variant
        )
        if use_riemannian:
            if args.optimizer == 'adam':
                print(
                    "[train] Using geoopt RiemannianAdam for CUSP (learnable curvatures); "
                    "Euclidean torch.optim.Adam destabilizes c/k. Use --euclidean_variant for a fully Euclidean model."
                )
            # L2 weight_decay on curvature reparams (isp_c / isp_k in newer geoopt) destabilizes training.
            _mid = _geoopt_stereographic_trainable_param_ids(model)
            wd_params, no_wd_params = [], []
            for _, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if id(p) in _mid:
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)
            if not args.euclidean_variant and not no_wd_params:
                print(
                    "[train] WARN: no Stereographic manifold params found for weight_decay=0; "
                    "check geoopt version / imports."
                )
            param_groups = []
            if wd_params:
                param_groups.append({"params": wd_params, "weight_decay": args.weight_decay})
            if no_wd_params:
                param_groups.append({"params": no_wd_params, "weight_decay": 0.0})
            if not param_groups:
                param_groups = [{"params": model.parameters(), "weight_decay": args.weight_decay}]
            optimizer = geoopt.optim.RiemannianAdam(
                param_groups, lr=args.lr, stabilize=50
            )
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Define loss functions based on task
        export_one = None
        if args.export_dir:
            os.makedirs(args.export_dir, exist_ok=True)
            export_one = os.path.join(args.export_dir, "run_seed_%s.json" % current_seed)

        if args.task == 'node_classification':
            criterion = F.nll_loss
            best_metric = train_node_classification(
                model, data, optimizer, scheduler, args, export_path=export_one
            )
        elif args.task == 'link_prediction':
            criterion = F.binary_cross_entropy_with_logits
            best_metric = train_link_prediction(
                model, data, optimizer, scheduler, args, export_path=export_one
            )
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        metrics.append(best_metric)

    nruns = len(metrics)
    print(f"\nFinal Results over {nruns} runs:")

    if args.task == 'node_classification':
        def _mean_std(key, sub=None):
            vals = []
            for m in metrics:
                d = m if sub is None else m[sub]
                vals.append(d[key])
            return float(np.mean(vals)), float(np.std(vals))

        for label, k, sub in [
            ("Test Acc", "accuracy", "best_test_metrics"),
            ("Test F1 (macro)", "f1_macro", "best_test_metrics"),
            ("Test F1 (micro)", "f1_micro", "best_test_metrics"),
            ("Test F1 (weighted)", "f1_weighted", "best_test_metrics"),
        ]:
            mu, sd = _mean_std(k, sub)
            print(f"  {label}: {mu:.4f} ± {sd:.4f}")

        tt_mu, tt_sd = _mean_std("total_train_loop_sec", "timing")
        te_mu, te_sd = _mean_std("total_test_eval_sec", "timing")
        pe_mu, pe_sd = _mean_std("avg_sec_per_epoch", "timing")
        print(f"  Total train time (s): {tt_mu:.4f} ± {tt_sd:.4f}")
        print(f"  Avg time per epoch (s): {pe_mu:.4f} ± {pe_sd:.4f}")
        print(f"  Total test eval time (s): {te_mu:.4f} ± {te_sd:.4f}")

        if args.export_dir:
            hparams = export_hparams_dict(args, data_root)
            summary = {
                "hparams": hparams,
                "dataset": args.dataset,
                "task": args.task,
                "seeds": list(run_seeds),
                "aggregate": {
                    "test_accuracy": {"mean": _mean_std("accuracy", "best_test_metrics")[0], "std": _mean_std("accuracy", "best_test_metrics")[1]},
                    "test_f1_macro": {"mean": _mean_std("f1_macro", "best_test_metrics")[0], "std": _mean_std("f1_macro", "best_test_metrics")[1]},
                    "test_f1_micro": {"mean": _mean_std("f1_micro", "best_test_metrics")[0], "std": _mean_std("f1_micro", "best_test_metrics")[1]},
                    "test_f1_weighted": {"mean": _mean_std("f1_weighted", "best_test_metrics")[0], "std": _mean_std("f1_weighted", "best_test_metrics")[1]},
                    "total_train_loop_sec": {"mean": tt_mu, "std": tt_sd},
                    "avg_sec_per_epoch": {"mean": pe_mu, "std": pe_sd},
                    "total_test_eval_sec": {"mean": te_mu, "std": te_sd},
                },
                "per_run": metrics,
            }
            out_dir = args.export_dir
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
                json.dump(hparams, f, indent=2, ensure_ascii=False)
            print("Wrote summary to %s" % os.path.join(out_dir, "summary.json"))
            print("Wrote hparams to %s" % os.path.join(out_dir, "experiment_config.json"))

    elif args.task == 'link_prediction':
        aucs = [m["best_auc"] for m in metrics]
        aps = [m["best_ap"] for m in metrics]
        accs = [m["best_test_metrics"]["accuracy"] for m in metrics]
        f1ma = [m["best_test_metrics"]["f1_macro"] for m in metrics]
        f1mi = [m["best_test_metrics"]["f1_micro"] for m in metrics]
        f1w = [m["best_test_metrics"]["f1_weighted"] for m in metrics]
        print(f"  Best AUC: {float(np.mean(aucs)):.4f} ± {float(np.std(aucs)):.4f}")
        print(f"  Best AP: {float(np.mean(aps)):.4f} ± {float(np.std(aps)):.4f}")
        print(
            f"  Test Acc (@ best AUC): {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}"
        )
        print(
            f"  Test F1 macro/micro/weighted (@ best AUC): {float(np.mean(f1ma)):.4f} ± {float(np.std(f1ma)):.4f} / "
            f"{float(np.mean(f1mi)):.4f} ± {float(np.std(f1mi)):.4f} / {float(np.mean(f1w)):.4f} ± {float(np.std(f1w)):.4f}"
        )
        tt_mu, tt_sd = float(np.mean([m["timing"]["total_train_loop_sec"] for m in metrics])), float(np.std([m["timing"]["total_train_loop_sec"] for m in metrics]))
        te_mu, te_sd = float(np.mean([m["timing"]["total_test_eval_sec"] for m in metrics])), float(np.std([m["timing"]["total_test_eval_sec"] for m in metrics]))
        pe_mu, pe_sd = float(np.mean([m["timing"]["avg_sec_per_epoch"] for m in metrics])), float(np.std([m["timing"]["avg_sec_per_epoch"] for m in metrics]))
        print(f"  Total train time (s): {tt_mu:.4f} ± {tt_sd:.4f}")
        print(f"  Avg time per epoch (s): {pe_mu:.4f} ± {pe_sd:.4f}")
        print(f"  Total test eval time (s): {te_mu:.4f} ± {te_sd:.4f}")
        if args.export_dir:
            hparams = export_hparams_dict(args, data_root)
            summary = {
                "hparams": hparams,
                "dataset": args.dataset,
                "task": args.task,
                "seeds": list(run_seeds),
                "aggregate": {
                    "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
                    "ap": {"mean": float(np.mean(aps)), "std": float(np.std(aps))},
                    "test_accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
                    "test_f1_macro": {"mean": float(np.mean(f1ma)), "std": float(np.std(f1ma))},
                    "test_f1_micro": {"mean": float(np.mean(f1mi)), "std": float(np.std(f1mi))},
                    "test_f1_weighted": {"mean": float(np.mean(f1w)), "std": float(np.std(f1w))},
                    "total_train_loop_sec": {"mean": tt_mu, "std": tt_sd},
                    "avg_sec_per_epoch": {"mean": pe_mu, "std": pe_sd},
                    "total_test_eval_sec": {"mean": te_mu, "std": te_sd},
                },
                "per_run": metrics,
            }
            out_dir = args.export_dir
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
                json.dump(hparams, f, indent=2, ensure_ascii=False)
            print("Wrote summary to %s" % os.path.join(out_dir, "summary.json"))
            print("Wrote hparams to %s" % os.path.join(out_dir, "experiment_config.json"))

def evaluate_node_classification(model, data):
    """Train/val/test accuracy and macro/micro/weighted F1."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
    train_m = nc_split_metrics(preds, labels, data.train_mask)
    val_m = nc_split_metrics(preds, labels, data.val_mask)
    test_m = nc_split_metrics(preds, labels, data.test_mask)
    return {"train": train_m, "val": val_m, "test": test_m}


def train_node_classification(model, data, optimizer, scheduler, args, export_path=None):
    """Node classification: best checkpoint by val weighted F1; returns metrics, timing, curves."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    best_val_f1 = -1.0
    best_test_metrics = None
    patience_ctr = 0

    curves = {
        "epoch": [],
        "loss": [],
        "val_f1_weighted": [],
        "test_accuracy": [],
        "test_f1_macro": [],
        "test_f1_micro": [],
        "test_f1_weighted": [],
    }
    total_eval_sec = 0.0
    total_train_step_sec = 0.0
    epochs_ran = 0
    training_failed = None

    t_loop0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        t_step0 = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(
            out[data.train_mask], data.y[data.train_mask], ignore_index=-1
        )
        loss_item = float(loss.detach().cpu().item())
        if not math.isfinite(loss_item):
            epochs_ran = epoch - 1
            training_failed = {
                "reason": "non_finite_loss",
                "epoch": int(epoch),
                "loss": loss_item,
                "hint": "Lower --learning_rate, try --euclidean_variant, or inspect data/features for NaN/Inf.",
            }
            print(
                "ERROR: loss is NaN/Inf at epoch %d (value=%s). Stopping. %s"
                % (epoch, repr(loss_item), training_failed["hint"])
            )
            break
        loss.backward()
        _gclip = 0.25 if args.dataset in TABLE2GRAPH_NAMES else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_gclip)
        optimizer.step()
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_train_step_sec += time.perf_counter() - t_step0

        t_eval0 = time.perf_counter()
        metrics = evaluate_node_classification(model, data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t_eval0
        total_eval_sec += eval_sec

        val_f1 = metrics["val"]["f1_weighted"]
        tm = metrics["test"]
        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_test_metrics = dict(tm)
            patience_ctr = 0
        else:
            patience_ctr += 1

        curves["epoch"].append(epoch)
        curves["loss"].append(float(loss.item()))
        curves["val_f1_weighted"].append(val_f1)
        curves["test_accuracy"].append(tm["accuracy"])
        curves["test_f1_macro"].append(tm["f1_macro"])
        curves["test_f1_micro"].append(tm["f1_micro"])
        curves["test_f1_weighted"].append(tm["f1_weighted"])

        print(
            "Epoch: %03d, Loss: %.4f, Val F1(w): %.4f, Test Acc: %.4f, Test F1 m/M/w: %.4f/%.4f/%.4f, Eval: %.4fs"
            % (
                epoch,
                loss.item(),
                val_f1,
                tm["accuracy"],
                tm["f1_macro"],
                tm["f1_micro"],
                tm["f1_weighted"],
                eval_sec,
            )
        )

        if args.patience > 0 and patience_ctr >= args.patience:
            print("Early stopping at epoch %d (no val F1 improvement for %d epochs)." % (epoch, args.patience))
            break

    wall_train_loop_sec = time.perf_counter() - t_loop0
    avg_sec_per_epoch = wall_train_loop_sec / max(epochs_ran, 1)

    if best_test_metrics is None:
        best_test_metrics = {k: 0.0 for k in ("accuracy", "f1_macro", "f1_micro", "f1_weighted")}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_test0 = time.perf_counter()
    _ = evaluate_node_classification(model, data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_test_eval_sec = time.perf_counter() - t_test0

    print(
        "Best Val F1 (weighted): %.4f | Best test @ that epoch: Acc %.4f, F1 macro/micro/w: %.4f/%.4f/%.4f"
        % (
            max(best_val_f1, 0.0),
            best_test_metrics["accuracy"],
            best_test_metrics["f1_macro"],
            best_test_metrics["f1_micro"],
            best_test_metrics["f1_weighted"],
        )
    )
    print(
        "Timing (NC): train_loop_wall=%.4fs, train_step_sum=%.4fs, eval_in_loop=%.4fs, avg/epoch=%.4fs, test_eval_once=%.4fs"
        % (wall_train_loop_sec, total_train_step_sec, total_eval_sec, avg_sec_per_epoch, total_test_eval_sec)
    )

    filter_weights = model.get_filter_weights()
    if filter_weights is not None:
        print("\nFilter Weights (epsilon):")
        print(filter_weights)

    component_weights = model.get_component_weights()
    if component_weights is not None:
        print("\nComponent Weights (theta):")
        for idx, theta in enumerate(component_weights):
            print("Theta %d: %s" % (idx, theta))

    print("\nLearned Curvatures:")
    final_curvatures = model.get_curvatures()
    for key, value in final_curvatures.items():
        print("%s: %s" % (key, value))

    out = {
        "best_val_f1_weighted": float(best_val_f1),
        "best_test_metrics": {k: float(v) for k, v in best_test_metrics.items()},
        "timing": {
            "total_train_loop_sec": float(wall_train_loop_sec),
            "total_train_step_sec": float(total_train_step_sec),
            "total_eval_during_train_sec": float(total_eval_sec),
            "avg_sec_per_epoch": float(avg_sec_per_epoch),
            "total_test_eval_sec": float(total_test_eval_sec),
        },
        "curves": curves,
        "epochs_ran": int(epochs_ran),
    }
    if training_failed is not None:
        out["training_failed"] = training_failed
    if export_path:
        serial = {
            "best_val_f1_weighted": out["best_val_f1_weighted"],
            "best_test_metrics": out["best_test_metrics"],
            "timing": out["timing"],
            "curves": out["curves"],
            "epochs_ran": out["epochs_ran"],
        }
        if training_failed is not None:
            serial["training_failed"] = training_failed
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2, ensure_ascii=False)
        print("Wrote run log to %s" % export_path)
    return out



def train_link_prediction(model, data, optimizer, scheduler, args, export_path=None):
    """Link prediction: early stopping on test AUC (legacy behavior). Returns AUC/AP, timing, curves."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    best_auc = -1.0
    best_ap = 0.0
    best_test_metrics = None
    patience_ctr = 0

    curves = {
        "epoch": [],
        "loss": [],
        "test_auc": [],
        "test_ap": [],
        "test_accuracy": [],
        "test_f1_macro": [],
        "test_f1_micro": [],
        "test_f1_weighted": [],
    }
    total_eval_sec = 0.0
    total_train_step_sec = 0.0
    epochs_ran = 0

    train_neg_edge_index = sample_train_neg_edges_lp(
        data, num_neg_directed=data.train_pos_edge_index.size(1)
    )

    t_loop0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        t_step0 = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        if args.model == "cusp":
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            z = model.encode(data.x, data.train_pos_edge_index)

        loss = link_prediction_loss(model, z, data.train_pos_edge_index, train_neg_edge_index)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_train_step_sec += time.perf_counter() - t_step0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_eval0 = time.perf_counter()
        auc, ap, tm = evaluate_link_prediction(args, model, data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t_eval0
        total_eval_sec += eval_sec
        if auc > best_auc + args.min_delta:
            best_auc = auc
            best_ap = ap
            best_test_metrics = dict(tm)
            patience_ctr = 0
        else:
            patience_ctr += 1

        curves["epoch"].append(epoch)
        curves["loss"].append(float(loss.item()))
        curves["test_auc"].append(float(auc))
        curves["test_ap"].append(float(ap))
        curves["test_accuracy"].append(tm["accuracy"])
        curves["test_f1_macro"].append(tm["f1_macro"])
        curves["test_f1_micro"].append(tm["f1_micro"])
        curves["test_f1_weighted"].append(tm["f1_weighted"])

        print(
            "Epoch: %03d, Loss: %.4f, AUC: %.4f, AP: %.4f, Acc: %.4f, F1 m/M/w: %.4f/%.4f/%.4f, Eval: %.4fs"
            % (
                epoch,
                loss.item(),
                auc,
                ap,
                tm["accuracy"],
                tm["f1_macro"],
                tm["f1_micro"],
                tm["f1_weighted"],
                eval_sec,
            )
        )

        if args.patience > 0 and patience_ctr >= args.patience:
            print("Early stopping at epoch %d (no test AUC improvement for %d epochs)." % (epoch, args.patience))
            break

    wall_train_loop_sec = time.perf_counter() - t_loop0
    avg_sec_per_epoch = wall_train_loop_sec / max(epochs_ran, 1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_test0 = time.perf_counter()
    _ = evaluate_link_prediction(args, model, data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_test_eval_sec = time.perf_counter() - t_test0

    if best_test_metrics is None:
        best_test_metrics = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
        }
    print(
        "Best AUC: %.4f, Best AP: %.4f | @ that checkpoint — Test Acc: %.4f, F1 macro/micro/w: %.4f/%.4f/%.4f"
        % (
            best_auc,
            best_ap,
            best_test_metrics["accuracy"],
            best_test_metrics["f1_macro"],
            best_test_metrics["f1_micro"],
            best_test_metrics["f1_weighted"],
        )
    )
    print(
        "Timing (LP): train_loop_wall=%.4fs, train_step_sum=%.4fs, eval_in_loop=%.4fs, avg/epoch=%.4fs, test_eval_once=%.4fs"
        % (wall_train_loop_sec, total_train_step_sec, total_eval_sec, avg_sec_per_epoch, total_test_eval_sec)
    )

    filter_weights = model.get_filter_weights()
    if filter_weights is not None:
        print("\nFilter Weights (epsilon):")
        print(filter_weights)

    component_weights = model.get_component_weights()
    if component_weights is not None:
        print("\nComponent Weights (theta):")
        for idx, theta in enumerate(component_weights):
            print("Theta %d: %s" % (idx, theta))

    print("\nLearned Curvatures:")
    final_curvatures = model.get_curvatures()
    for key, value in final_curvatures.items():
        print("%s: %s" % (key, value))

    out = {
        "best_auc": float(best_auc),
        "best_ap": float(best_ap),
        "best_test_metrics": {k: float(v) for k, v in best_test_metrics.items()},
        "timing": {
            "total_train_loop_sec": float(wall_train_loop_sec),
            "total_train_step_sec": float(total_train_step_sec),
            "total_eval_during_train_sec": float(total_eval_sec),
            "avg_sec_per_epoch": float(avg_sec_per_epoch),
            "total_test_eval_sec": float(total_test_eval_sec),
        },
        "curves": curves,
        "epochs_ran": int(epochs_ran),
    }
    if export_path:
        serial = {
            "best_auc": out["best_auc"],
            "best_ap": out["best_ap"],
            "best_test_metrics": out["best_test_metrics"],
            "timing": out["timing"],
            "curves": out["curves"],
            "epochs_ran": out["epochs_ran"],
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2, ensure_ascii=False)
        print("Wrote run log to %s" % export_path)
    return out

def link_prediction_loss(model, z, pos_edge_index, neg_edge_index):
    """
    Compute link prediction loss for both positive and negative edges using the inner product decoder.
    """
    # Positive edge loss
    pos_logits = model.decode(z, pos_edge_index)
    pos_labels = torch.ones(pos_logits.size(0), device=pos_logits.device)

    # Negative edge loss
    neg_logits = model.decode(z, neg_edge_index)
    neg_labels = torch.zeros(neg_logits.size(0), device=neg_logits.device)

    # Concatenate positive and negative logits and labels
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def sample_train_neg_edges_lp(data, num_neg_directed, method="sparse"):
    """
    Train negatives via PyG negative_sampling (sparse, directed), not dense train_neg_adj_mask.nonzero (OOM on large N).
    Excludes train+val+test positive edges. CPU sampling avoids PyG force_undirected dtype bugs on some torch builds.
    Loops until enough directed negatives match train_pos_edge_index.size(1).
    """
    if num_neg_directed < 1:
        raise ValueError("num_neg_directed must be positive, got %d" % num_neg_directed)
    pos_parts = [data.train_pos_edge_index]
    if getattr(data, "val_pos_edge_index", None) is not None:
        pos_parts.append(data.val_pos_edge_index)
    if getattr(data, "test_pos_edge_index", None) is not None:
        pos_parts.append(data.test_pos_edge_index)
    pos_edge_index = torch.cat(pos_parts, dim=1)
    out_device = pos_edge_index.device
    pos_edge_index_cpu = pos_edge_index.cpu().long()

    chunks = []
    need_directed = num_neg_directed
    rounds = 0
    max_rounds = 128
    while need_directed > 0:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                "negative_sampling could not supply %d directed negatives after %d rounds"
                % (num_neg_directed, max_rounds)
            )
        chunk = negative_sampling(
            pos_edge_index_cpu,
            num_nodes=int(data.num_nodes),
            num_neg_samples=max(1, need_directed),
            method=method,
            force_undirected=False,
        )
        chunk = chunk.to(device=out_device, dtype=torch.long)
        if chunk.numel() == 0:
            raise RuntimeError(
                "negative_sampling returned empty tensor; graph may be too dense for link prediction."
            )
        chunks.append(chunk)
        need_directed -= chunk.size(1)

    neg_edge_index = torch.cat(chunks, dim=1)
    if neg_edge_index.size(1) > num_neg_directed:
        neg_edge_index = neg_edge_index[:, :num_neg_directed]
    return neg_edge_index

def evaluate_link_prediction(args, model, data):
    """
    Evaluation function for Link Prediction, handling both CUSP and baseline models.
    """
    model.eval()
    with torch.no_grad():
        if args.model == 'cusp':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            # Baseline models (GCN, GAT, SAGE) don't use kappa
            z = model.encode(data.x, data.train_pos_edge_index)

        auc, ap, test_cls_metrics = model.test(
            z, data.test_pos_edge_index, data.test_neg_edge_index
        )
    return auc, ap, test_cls_metrics

if __name__ == '__main__':
    main()