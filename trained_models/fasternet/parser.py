from argparse import ArgumentParser
from .utils.utils import *


def parse_fasternet_args(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, default="cfg/fasternet_t0.yaml")
    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        default=None,
        help="Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.",
    )
    parser.add_argument(
        "-d", "--dev", type=int, default=0, help="fast_dev_run for debug"
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    parser.add_argument("-b", "--batch_size", type=int, default=2048)
    parser.add_argument(
        "-e",
        "--batch_size_eva",
        type=int,
        default=1000,
        help="batch_size for evaluation",
    )
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--data_dir", type=str, default="../../data/imagenet")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--pconv_fw_type",
        type=str,
        default="split_cat",
        help="use 'split_cat' for training/inference and 'slicing' only for inference",
    )
    parser.add_argument(
        "--measure_latency", action="store_true", help="measure latency or throughput"
    )
    parser.add_argument("--test_phase", action="store_true")
    parser.add_argument("--fuse_conv_bn", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="fasternet")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_save_dir", type=str, default="./")
    parser.add_argument(
        "--pl_ckpt_2_torch_pth",
        action="store_true",
        help="convert pl .ckpt file to torch .pth file, and then exit",
    )

    args = parser.parse_args(raw_args)
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)
    return args
