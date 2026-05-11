import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


# utils/train_utils.py (get_dir_name 函数)

def get_dir_name(models_dir, args):
    # 1. 构造一个包含关键参数的字符串
    params_str = (
        f"{args.task}_"
        f"{args.dataset}_"
        f"{args.model}_"
        f"lr{args.lr}_"
        f"dim{args.dim}_"
        f"drp{args.dropout}"
    ).replace('.', 'p') # 将小数点替换为p，防止目录名非法
    
    # 2. 检查目录是否存在并生成唯一的运行编号
    
    run_num = 1
    # 寻找以该参数字符串开头的现有目录
    while os.path.isdir(os.path.join(models_dir, f"{params_str}_{run_num}")):
        run_num += 1
    
    # 3. 构造最终的目录名
    final_dir_name = f"logs/{params_str}_{run_num}"

    return final_dir_name



def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


