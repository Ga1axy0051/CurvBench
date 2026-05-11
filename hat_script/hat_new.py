import time
import argparse
import json
import csv
import numpy as np
import tensorflow as tf
import os
import random

# TF2 compat: v1 graph + Session
tf.compat.v1.disable_v2_behavior()

from models import SpMHGAT
from utils import process
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# Seeds 0..4 for run index < 5; else base_seed + r*10000
_RUN_SEEDS_FIXED = (0, 1, 2, 3, 4)

# Table2graph-style datasets: default raw feats + CE; optional standardize / class weights
TABLE_DATASETS = frozenset(
    {
        "carcinogenesis",
        "hepatitis_std",
        "hockey",
        "pte",
        "toxicology",
        "f1",
        "cs_phds_nc",
        "cs_phds_lp",
    }
)


def _resolve_feat_and_balance(args):
    """Resolve feat_preproc and whether to use balanced loss."""
    is_table = args.dataset in TABLE_DATASETS
    if args.feat_preproc is not None:
        fp = args.feat_preproc
    else:
        fp = "raw" if is_table else "rownorm"
    bl = bool(args.balanced_loss) and not bool(args.no_balanced_loss)
    return fp, bl


def _class_weights_balanced(y_train_2d, train_mask_1d, nb_classes):
    """sklearn balanced weights from train-mask nodes only."""
    m = np.asarray(train_mask_1d, dtype=bool).ravel()
    y_int = np.argmax(y_train_2d[m], axis=1)
    uniq = np.unique(y_int)
    cw_part = compute_class_weight("balanced", classes=uniq, y=y_int)
    full = np.ones(nb_classes, dtype=np.float32)
    for c, w in zip(uniq, cw_part):
        full[int(c)] = float(w)
    return full


def _run_seed_for_index(r, base_seed):
    if r < len(_RUN_SEEDS_FIXED):
        return _RUN_SEEDS_FIXED[r]
    return base_seed + r * 10000


def _apply_feat_symmetry_breaks(features, feat_sym_break_eps, feat_noise_std, run_seed):
    """Optional index-based shift on col 0, then optional Gaussian noise scaled per column."""
    X = np.asarray(features, dtype=np.float32)
    n = int(X.shape[0])
    fe = float(feat_sym_break_eps or 0.0)
    if fe > 0 and n > 0:
        X = np.array(X, copy=True)
        r = np.arange(n, dtype=np.float32) - 0.5 * float(n - 1)
        X[:, 0] = X[:, 0] + r * np.float32(fe)
        print("feat_sym_break_eps=%.8f: col0 index-centered shift applied" % fe)
    fn = float(feat_noise_std or 0.0)
    if fn > 0:
        rng = np.random.RandomState(int(run_seed))
        fd = X.astype(np.float64)
        sig = np.std(fd, axis=0, keepdims=True)
        sig = np.maximum(sig, 1e-12)
        noise = rng.standard_normal(size=fd.shape) * (fn * sig)
        X = (fd + noise).astype(np.float32)
        print("feat_noise_std=%.4f: Gaussian noise scaled by col std" % fn)
    return X


def _print_multiclass_acc_baselines(y_train, y_val, train_mask, val_mask, nb_classes):
    """Print random-guess baseline and majority-class share on train/val."""
    tm = np.asarray(train_mask, dtype=bool).ravel()
    vm = np.asarray(val_mask, dtype=bool).ravel()
    yt = np.asarray(y_train)
    yv = np.asarray(y_val)
    if yt.ndim == 3:
        yt, yv = yt[0], yv[0]
    yi_t = np.argmax(yt, axis=1)
    yi_v = np.argmax(yv, axis=1)
    bc_t = np.bincount(yi_t[tm], minlength=nb_classes)
    bc_v = np.bincount(yi_v[vm], minlength=nb_classes)
    nt, nv = int(bc_t.sum()), int(bc_v.sum())
    maj_t = float(bc_t.max() / max(nt, 1))
    maj_v = float(bc_v.max() / max(nv, 1))
    rand = 1.0 / float(nb_classes)
    print(
        "acc baseline: random 1/C=%.4f | train majority=%.4f | val majority=%.4f"
        % (rand, maj_t, maj_v)
    )
    print(
        "Note: collapsed predictions -> acc ~ majority share; use macro-F1 if imbalanced."
    )


def _f1_safe(y_true, y_pred, average):
    """sklearn f1; macro without forcing all class labels (see sklearn docs)."""
    try:
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    except TypeError:
        return f1_score(y_true, y_pred, average=average)


def parse_args():
    parser = argparse.ArgumentParser(description="run MHAT")
    parser.add_argument("-gpu", nargs="?", default="1", help="the ID for GPU")
    parser.add_argument("-dataset", nargs="?", default="cora", help="name of dataset")
    parser.add_argument("-lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("-l2", default=0.0001, type=float, help="l2")
    parser.add_argument("-units", default=8, type=int, help="dimension for hidden unit")
    parser.add_argument("-heads", default=1, type=int, help="number of multi-heads")
    parser.add_argument("-drop", default=0.2, type=float, help="drop out")
    parser.add_argument("-c", default=1, type=int, help="0: untrainable curvature; 1: trainable curvature")
    parser.add_argument(
        "--seed",
        type=int,
        default=66,
        help="r<5: seeds 0..4; r>=5: seed + r*10000",
    )
    parser.add_argument("--n_runs", type=int, default=5, help="repeat training for mean/std (NC metrics)")
    parser.add_argument("--nb_epochs", type=int, default=100000, help="max epochs")
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument("--min_delta_loss", type=float, default=1e-4, help="min val loss drop to count as improve")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="root log directory; default runs/<timestamp>",
    )
    parser.add_argument(
        "--feat_preproc",
        choices=("rownorm", "standardize_train", "raw"),
        default=None,
        help="raw=dense float32; rownorm=GCN row norm; standardize_train=train z-score (non-table default rownorm).",
    )
    parser.add_argument(
        "--balanced_loss",
        action="store_true",
        help="sklearn balanced class weights for CE (off by default)",
    )
    parser.add_argument(
        "--no_balanced_loss",
        action="store_true",
        help="if set with --balanced_loss, disables balanced loss",
    )
    parser.add_argument(
        "--debug_metrics",
        action="store_true",
        help="after epoch 0, print val prediction diversity",
    )
    parser.add_argument(
        "--feat_noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise N(0,(feat_noise_std*col_std)^2) per dim; 0 disables.",
    )
    parser.add_argument(
        "--feat_sym_break_eps",
        type=float,
        default=0.0,
        help="add (i-mean(i))*eps to feature col 0; 0 disables.",
    )
    return parser.parse_args()


def _set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def _build_config_dict(args, run_seed, batch_size, feat_preproc=None, balanced_loss=None):
    cfg = {
        "model": "SpMHGAT",
        "dataset": args.dataset,
        "gpu": args.gpu,
        "lr": args.lr,
        "l2": args.l2,
        "units": args.units,
        "heads": args.heads,
        "dropout": args.drop,
        "c_trainable_curvature": args.c,
        "activation": "elu",
        "nb_layers": 1,
        "n_heads_structure": [args.heads, 1],
        "batch_size": batch_size,
        "nb_epochs_max": args.nb_epochs,
        "patience": args.patience,
        "min_delta_loss": args.min_delta_loss,
        "seed": run_seed,
        "n_runs_total": args.n_runs,
        "task": "node_classification",
    }
    if feat_preproc is not None:
        cfg["feat_preproc"] = feat_preproc
    if balanced_loss is not None:
        cfg["balanced_loss"] = balanced_loss
    fn = float(getattr(args, "feat_noise_std", 0.0) or 0.0)
    if fn > 0:
        cfg["feat_noise_std"] = fn
    fe = float(getattr(args, "feat_sym_break_eps", 0.0) or 0.0)
    if fe > 0:
        cfg["feat_sym_break_eps"] = fe
    return cfg


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_history_csv(path, history):
    """Write history CSV; test_* filled when val improves."""
    keys = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "test_f1_macro",
        "test_f1_micro",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(history["epoch"])
        for i in range(n):
            w.writerow([history[k][i] for k in keys])


def run_single_training(args, run_seed, log_dir):
    """One training run; writes config/history/results under log_dir."""
    _set_all_seeds(run_seed)
    os.makedirs(log_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf_cfg = tf.compat.v1.ConfigProto()
    tf_cfg.gpu_options.allow_growth = True
    tf_cfg.allow_soft_placement = True

    batch_size = 1
    nb_epochs = args.nb_epochs
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2
    hid_units = [args.units]
    n_heads = [args.heads, 1]
    drop_out = args.drop
    nonlinearity = tf.nn.elu
    model = SpMHGAT
    c = args.c
    min_delta_loss = args.min_delta_loss

    feat_preproc, use_balanced = _resolve_feat_and_balance(args)
    cfg = _build_config_dict(
        args, run_seed, batch_size, feat_preproc, use_balanced
    )
    _save_json(os.path.join(log_dir, "config.json"), cfg)

    print("Dataset:", args.dataset)
    print("log_dir:", log_dir, "| seed:", run_seed)
    print("lr:", lr, "l2:", l2_coef, "units:", hid_units, "heads:", n_heads)
    print(
        "feat_preproc: %s | balanced_loss: %s"
        % (feat_preproc, use_balanced)
    )

    sparse = True
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(
        args.dataset
    )
    if feat_preproc == "standardize_train":
        features = process.preprocess_features_train_standardize(features, train_mask)
    elif feat_preproc == "rownorm":
        features, _ = process.preprocess_features(features)
    else:
        features = process.preprocess_features_raw_dense(features)

    features = process.maybe_exptable2graph_sanitize_nonfinite(
        features,
        args.dataset in TABLE_DATASETS,
        tag="%s/features" % args.dataset,
    )

    features = _apply_feat_symmetry_breaks(
        features,
        getattr(args, "feat_sym_break_eps", 0.0),
        getattr(args, "feat_noise_std", 0.0),
        run_seed,
    )

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]
    if nb_classes > 2:
        _print_multiclass_acc_baselines(
            y_train, y_val, train_mask, val_mask, nb_classes
        )

    if use_balanced:
        cw_np = _class_weights_balanced(y_train, train_mask, nb_classes)
        print(
            "balanced class weights: min=%.4f mean=%.4f max=%.4f"
            % (float(cw_np.min()), float(cw_np.mean()), float(cw_np.max()))
        )
    else:
        cw_np = np.ones(nb_classes, dtype=np.float32)

    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    if sparse:
        biases = process.preprocess_adj_bias(adj)
    else:
        adj = adj.todense()
        adj = adj[np.newaxis]
        biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1_macro": [],
        "test_f1_micro": [],
    }

    ts_loss = float("nan")
    ts_acc = float("nan")
    ts_macro = float("nan")
    ts_micro = float("nan")
    ts_step = 0
    best_val_loss = np.inf
    best_val_acc = 0.0
    best_epoch = -1
    curr_step = 0
    last_val_sec = 0.0
    last_test_sec = 0.0
    sum_val_sec = 0.0
    sum_test_sec = 0.0
    test_run_count = 0

    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.name_scope("input"):
            ftr_in = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size)
            )
            if sparse:
                bias_in = tf.compat.v1.sparse_placeholder(dtype=tf.float32)
            else:
                bias_in = tf.compat.v1.placeholder(
                    dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes)
                )
            # float32 one-hot for CE / argmax
            lbl_in = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(batch_size, nb_nodes, nb_classes)
            )
            msk_in = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=(batch_size, nb_nodes)
            )
            attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
            is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())

        logits, emb, curvature = model.inference(
            ftr_in,
            nb_classes,
            nb_nodes,
            is_train,
            attn_drop,
            ffd_drop,
            bias_mat=bias_in,
            hid_units=hid_units,
            n_heads=n_heads,
            activation=nonlinearity,
            c=c,
        )
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        if use_balanced:
            cw_const = tf.constant(cw_np, dtype=tf.float32)
            loss = model.masked_softmax_cross_entropy_weighted(
                log_resh, lab_resh, msk_resh, cw_const
            )
        else:
            loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        pred_all = tf.cast(tf.argmax(log_resh, 1), dtype=tf.int32)
        real_all = tf.cast(tf.argmax(lab_resh, 1), dtype=tf.int32)
        train_op = model.my_training(loss, lr, l2_coef)

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer(),
        )

        with tf.compat.v1.Session(config=tf_cfg) as sess:
            sess.run(init_op)

            t_train_start = time.perf_counter()
            epoch = 0
            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]
                while tr_step * batch_size < tr_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[tr_step * batch_size : (tr_step + 1) * batch_size]

                    # train step uses dropout
                    _, loss_value_tr, train_emb, curvature_this = sess.run(
                        [train_op, loss, emb, curvature],
                        feed_dict={
                            ftr_in: features[tr_step * batch_size : (tr_step + 1) * batch_size],
                            bias_in: bbias,
                            lbl_in: y_train[tr_step * batch_size : (tr_step + 1) * batch_size],
                            msk_in: train_mask[tr_step * batch_size : (tr_step + 1) * batch_size],
                            is_train: True,
                            attn_drop: drop_out,
                            ffd_drop: drop_out,
                        },
                    )
                    tr_step += 1

                # train metrics at dropout=0 (match val/test)
                if sparse:
                    _bb_tr_eval = biases
                else:
                    _bb_tr_eval = biases[0:batch_size]
                loss_tr_eval, acc_tr_eval = sess.run(
                    [loss, accuracy],
                    feed_dict={
                        ftr_in: features[0:batch_size],
                        bias_in: _bb_tr_eval,
                        lbl_in: y_train[0:batch_size],
                        msk_in: train_mask[0:batch_size],
                        is_train: False,
                        attn_drop: 0.0,
                        ffd_drop: 0.0,
                    },
                )

                vl_step = 0
                vl_size = features.shape[0]
                val_loss_avg = 0
                val_acc_avg = 0

                t_val0 = time.perf_counter()
                while vl_step * batch_size < vl_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[vl_step * batch_size : (vl_step + 1) * batch_size]
                    loss_value_vl, acc_vl = sess.run(
                        [loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step * batch_size : (vl_step + 1) * batch_size],
                            bias_in: bbias,
                            lbl_in: y_val[vl_step * batch_size : (vl_step + 1) * batch_size],
                            msk_in: val_mask[vl_step * batch_size : (vl_step + 1) * batch_size],
                            is_train: False,
                            attn_drop: 0.0,
                            ffd_drop: 0.0,
                        },
                    )
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1
                last_val_sec = time.perf_counter() - t_val0
                sum_val_sec += last_val_sec

                tl = float(loss_tr_eval)
                ta = float(acc_tr_eval)
                vl = val_loss_avg / vl_step
                va = val_acc_avg / vl_step

                if epoch == 0 and getattr(args, "debug_metrics", False):
                    _bb0 = biases if sparse else biases[0:batch_size]
                    pred_v, real_v = sess.run(
                        [pred_all, real_all],
                        feed_dict={
                            ftr_in: features[0:batch_size],
                            bias_in: _bb0,
                            lbl_in: y_val[0:batch_size],
                            msk_in: val_mask[0:batch_size],
                            is_train: False,
                            attn_drop: 0.0,
                            ffd_drop: 0.0,
                        },
                    )
                    m = val_mask[0].astype(bool)
                    pv, rv = np.asarray(pred_v[m]), np.asarray(real_v[m])
                    if len(pv) > 0:
                        u_pred = len(np.unique(pv))
                        mode_cnt = int(np.bincount(pv, minlength=nb_classes).max())
                        print(
                            "  [debug_metrics] val labeled=%d | pred n_classes=%d | "
                            "mode_frac=%.4f (1.0 ~ collapsed)"
                            % (len(pv), u_pred, mode_cnt / float(len(pv)))
                        )

                val_loss_epoch = vl
                val_acc_epoch = va
                improved = val_loss_epoch < (best_val_loss - min_delta_loss)

                test_loss_e = float("nan")
                test_acc_e = float("nan")
                test_macro_e = float("nan")
                test_micro_e = float("nan")

                if improved:
                    best_val_loss = val_loss_epoch
                    best_val_acc = val_acc_epoch
                    best_epoch = epoch
                    curr_step = 0

                    ts_size = features.shape[0]
                    ts_step = 0
                    ts_loss = 0.0
                    ts_acc = 0.0
                    ts_macro = 0.0
                    ts_micro = 0.0
                    t_test0 = time.perf_counter()
                    while ts_step * batch_size < ts_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[ts_step * batch_size : (ts_step + 1) * batch_size]
                        loss_value_ts, acc_ts, test_emb, real_y, pred_y = sess.run(
                            [loss, accuracy, emb, real_all, pred_all],
                            feed_dict={
                                ftr_in: features[ts_step * batch_size : (ts_step + 1) * batch_size],
                                bias_in: bbias,
                                lbl_in: y_test[ts_step * batch_size : (ts_step + 1) * batch_size],
                                msk_in: test_mask[ts_step * batch_size : (ts_step + 1) * batch_size],
                                is_train: False,
                                attn_drop: 0.0,
                                ffd_drop: 0.0,
                            },
                        )
                        ts_loss += loss_value_ts
                        ts_acc += acc_ts
                        ts_step += 1
                        m = test_mask[0]
                        ts_macro += _f1_safe(real_y[m], pred_y[m], "macro")
                        ts_micro += _f1_safe(real_y[m], pred_y[m], "micro")
                    last_test_sec = time.perf_counter() - t_test0
                    sum_test_sec += last_test_sec
                    test_run_count += 1
                    test_loss_e = float(ts_loss / ts_step)
                    test_acc_e = float(ts_acc / ts_step)
                    test_macro_e = float(ts_macro / ts_step)
                    test_micro_e = float(ts_micro / ts_step)
                else:
                    curr_step += 1

                history["epoch"].append(epoch)
                history["train_loss"].append(float(tl))
                history["train_acc"].append(float(ta))
                history["val_loss"].append(float(vl))
                history["val_acc"].append(float(va))
                history["test_loss"].append(test_loss_e)
                history["test_acc"].append(test_acc_e)
                history["test_f1_macro"].append(test_macro_e)
                history["test_f1_micro"].append(test_micro_e)

                print(
                    epoch,
                    "Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | Val eval time: %.4fs"
                    % (tl, ta, vl, va, last_val_sec),
                )
                if improved:
                    print(
                        "  Test (val improved): loss=%.5f acc=%.5f macro=%.5f micro=%.5f | time %.4fs"
                        % (
                            test_loss_e,
                            test_acc_e,
                            test_macro_e,
                            test_micro_e,
                            last_test_sec,
                        )
                    )

                if not improved and curr_step == patience:
                    print(
                        "Early stop! Best epoch:",
                        best_epoch,
                        ", Best val loss:",
                        best_val_loss,
                        ", val acc:",
                        best_val_acc,
                    )
                    break

            total_train_sec = time.perf_counter() - t_train_start
            epochs_run = epoch + 1
            avg_val_sec = (sum_val_sec / epochs_run) if epochs_run > 0 else 0.0
            avg_epoch_wall_sec = (total_train_sec / epochs_run) if epochs_run > 0 else 0.0

            if ts_step > 0:
                print(
                    "Final test (best val checkpoint): loss=%.5f acc=%.5f macro=%.5f micro=%.5f"
                    % (
                        ts_loss / ts_step,
                        ts_acc / ts_step,
                        ts_macro / ts_step,
                        ts_micro / ts_step,
                    )
                )
                print(
                    "  last Val eval time: %.4fs | last Test eval time: %.4fs"
                    % (last_val_sec, last_test_sec)
                )
            print("--- Timing summary ---")
            print(
                "Total wall-clock training time: %.2fs (%d epochs)" % (total_train_sec, epochs_run)
            )
            print(
                "Avg wall-clock per epoch (total_time/epochs): %.4fs"
                % avg_epoch_wall_sec
            )
            print(
                "Cumulative Val eval time: %.2fs (avg %.4fs / epoch)"
                % (sum_val_sec, avg_val_sec)
            )
            print(
                "Cumulative Test eval time: %.2fs (%d test runs, only when val improved)"
                % (sum_test_sec, test_run_count)
            )

            sess.close()

    _save_history_csv(os.path.join(log_dir, "history.csv"), history)

    test_loss_f = float(ts_loss / ts_step) if ts_step > 0 else float("nan")
    test_acc_f = float(ts_acc / ts_step) if ts_step > 0 else float("nan")
    macro_f = float(ts_macro / ts_step) if ts_step > 0 else float("nan")
    micro_f = float(ts_micro / ts_step) if ts_step > 0 else float("nan")

    avg_test_sec = (
        float(sum_test_sec / test_run_count) if test_run_count > 0 else float("nan")
    )
    results = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss < np.inf else None,
        "best_val_acc": float(best_val_acc),
        "test_loss": test_loss_f,
        "test_acc": test_acc_f,
        "test_f1_macro": macro_f,
        "test_f1_micro": micro_f,
        "epochs_run": int(epochs_run),
        "total_train_wall_sec": float(total_train_sec),
        "avg_wall_sec_per_epoch": float(avg_epoch_wall_sec),
        "cumulative_val_eval_sec": float(sum_val_sec),
        "cumulative_test_eval_sec": float(sum_test_sec),
        "test_run_count": int(test_run_count),
        "time_per_epoch_sec": float(avg_epoch_wall_sec),
        "total_test_eval_sec": float(sum_test_sec),
        "avg_test_eval_sec_per_val_improve": avg_test_sec,
    }
    _save_json(os.path.join(log_dir, "results.json"), results)

    return {
        "test_acc": test_acc_f,
        "test_f1_macro": macro_f,
        "test_f1_micro": micro_f,
        "test_loss": test_loss_f,
        "total_train_wall_sec": float(total_train_sec),
        "time_per_epoch_sec": float(avg_epoch_wall_sec),
        "total_test_eval_sec": float(sum_test_sec),
        "results_path": os.path.join(log_dir, "results.json"),
        "history_path": os.path.join(log_dir, "history.csv"),
    }


def main(args):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_root = args.log_dir or os.path.join("runs", stamp)
    os.makedirs(log_root, exist_ok=True)

    fp_m, bl_m = _resolve_feat_and_balance(args)
    master_cfg = _build_config_dict(args, args.seed, 1, fp_m, bl_m)
    master_cfg.pop("seed", None)
    master_cfg["seed_base"] = args.seed
    master_cfg["log_root"] = os.path.abspath(log_root)
    master_cfg["timestamp"] = stamp
    master_cfg["run_seeds"] = [
        _run_seed_for_index(r, args.seed) for r in range(args.n_runs)
    ]
    master_cfg["run_seed_policy"] = "r<5 -> 0,1,2,3,4; r>=5 -> seed_base + r*10000"
    _save_json(os.path.join(log_root, "master_config.json"), master_cfg)

    all_rows = []
    for r in range(args.n_runs):
        run_seed = _run_seed_for_index(r, args.seed)
        run_dir = os.path.join(log_root, "run_%02d_seed_%d" % (r, run_seed))
        if args.n_runs > 1:
            tf.compat.v1.reset_default_graph()
        out = run_single_training(args, run_seed, run_dir)
        all_rows.append(
            {
                "run": r,
                "seed": run_seed,
                "test_acc": out["test_acc"],
                "test_f1_macro": out["test_f1_macro"],
                "test_f1_micro": out["test_f1_micro"],
                "test_loss": out["test_loss"],
                "total_train_wall_sec": out["total_train_wall_sec"],
                "time_per_epoch_sec": out["time_per_epoch_sec"],
                "total_test_eval_sec": out["total_test_eval_sec"],
            }
        )

    if args.n_runs > 1:
        accs = np.array([x["test_acc"] for x in all_rows], dtype=np.float64)
        macros = np.array([x["test_f1_macro"] for x in all_rows], dtype=np.float64)
        micros = np.array([x["test_f1_micro"] for x in all_rows], dtype=np.float64)
        train_t = np.array([x["total_train_wall_sec"] for x in all_rows], dtype=np.float64)
        tpe = np.array([x["time_per_epoch_sec"] for x in all_rows], dtype=np.float64)
        ttest = np.array([x["total_test_eval_sec"] for x in all_rows], dtype=np.float64)
        agg = {
            "n_runs": args.n_runs,
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs, ddof=1)),
            "test_f1_macro_mean": float(np.mean(macros)),
            "test_f1_macro_std": float(np.std(macros, ddof=1)),
            "test_f1_micro_mean": float(np.mean(micros)),
            "test_f1_micro_std": float(np.std(micros, ddof=1)),
            "total_train_wall_sec_mean": float(np.mean(train_t)),
            "total_train_wall_sec_std": float(np.std(train_t, ddof=1)),
            "time_per_epoch_sec_mean": float(np.mean(tpe)),
            "time_per_epoch_sec_std": float(np.std(tpe, ddof=1)),
            "total_test_eval_sec_mean": float(np.mean(ttest)),
            "total_test_eval_sec_std": float(np.std(ttest, ddof=1)),
            "runs": all_rows,
        }
        _save_json(os.path.join(log_root, "aggregate_results.json"), agg)
        print("=== Aggregate over %d runs ===" % args.n_runs)
        print(
            "test_acc: %.6f ± %.6f"
            % (agg["test_acc_mean"], agg["test_acc_std"])
        )
        print(
            "test_f1_macro: %.6f ± %.6f"
            % (agg["test_f1_macro_mean"], agg["test_f1_macro_std"])
        )
        print(
            "test_f1_micro: %.6f ± %.6f"
            % (agg["test_f1_micro_mean"], agg["test_f1_micro_std"])
        )
        print(
            "total_train_wall_sec: %.4f ± %.4f"
            % (agg["total_train_wall_sec_mean"], agg["total_train_wall_sec_std"])
        )
        print(
            "time_per_epoch_sec: %.6f ± %.6f"
            % (agg["time_per_epoch_sec_mean"], agg["time_per_epoch_sec_std"])
        )
        print(
            "total_test_eval_sec: %.4f ± %.4f"
            % (agg["total_test_eval_sec_mean"], agg["total_test_eval_sec_std"])
        )


if __name__ == "__main__":
    main(parse_args())
