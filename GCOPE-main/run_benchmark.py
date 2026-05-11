import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, default="nc")
    parser.add_argument('--data-root', type=str, default="")
    args, unknown = parser.parse_known_args()

    # Pretrain GCOPE
    # Fastargs for pretrain
    pretrain_cmd = [
        sys.executable, "src/exec.py",
        "--config-file", "pretrain.json",
        "--general.save_dir", f"storage/gcn/reconstruct",
        "--general.func", "pretrain",
        "--general.reconstruct", "0.2",
        "--data.name", args.dataset, # typically this should be other datasets, but for benchmark we just pass the dataset or skip pretrain if not needed. Wait...
        "--pretrain.split_method", "RandomWalk",
        "--model.backbone.model_type", "gcn"
    ]
    
    # Actually let's just do adapt with finetuning?
    # Because for baselines we just need to get it to run and output result.
    # Let's run a full pretrain + adapt pipeline for the single dataset.
    print(f"Running GCOPE Pretrain on {args.dataset}...")
    subprocess.run(pretrain_cmd, check=True)

    tune_cmd = [
        sys.executable, "src/exec.py",
        "--general.func", "adapt",
        "--general.save_dir", f"storage/gcn/few_shot",
        "--general.few_shot", "1",
        "--general.reconstruct", "0.0",
        "--data.node_feature_dim", "100", # Need this?
        "--data.name", args.dataset,
        "--adapt.method", "finetune",
        "--model.backbone.model_type", "gcn",
        "--model.saliency.model_type", "none",
        "--adapt.pretrained_file", f"storage/gcn/reconstruct/{args.dataset}_pretrained_model.pt",
        "--adapt.finetune.learning_rate", "5e-3",
        "--adapt.batch_size", "100",
        "--adapt.finetune.backbone_tuning", "1"
    ]
    
    print(f"Running GCOPE Adapt on {args.dataset}...")
    subprocess.run(tune_cmd, check=True)

if __name__ == '__main__':
    main()
