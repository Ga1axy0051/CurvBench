import torch
from configs import parse_pretrain_config, parse_adaption_config
from cores.pretrain_trainer import Pretrainer
from downstream.adapt_trainer import AdaptTrainer
from utils import set_seed
import argparse


def pretrain_main(remaining_argv=None):
    config = parse_pretrain_config(remaining_argv)
    set_seed(config.seed)

    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    trainer = Pretrainer(config)
    trainer.train()
    torch.cuda.empty_cache()


def transfer_main(remaining_argv=None):
    config = parse_adaption_config(remaining_argv)
    set_seed(config.seed)

    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    trainer = AdaptTrainer(config)
    trainer.train()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain or Adaption Command")
    parser.add_argument("--run_type", type=str, default="pretrain", choices=["pretrain", "adapt"])
    args, remaining_argv = parser.parse_known_args()
    if args.run_type == "pretrain":
        pretrain_main(remaining_argv)
    elif args.run_type == "adapt":
        transfer_main(remaining_argv)
    else:
        raise ValueError("Invalid run type")
