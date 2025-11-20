import hashlib
import logging
import os
import pathlib
import random
import subprocess

import numpy as np
import torch
from torch import backends, cuda

_ENABLE_APPLE_GPU = False


def seed(n):
    random.seed(n)
    np.random.seed(n)


def get_free_gpu():
    max_free = 0
    max_idx = 0

    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        .decode("utf-8")
        .split("\n")
    )
    for i, row in enumerate(rows[1:-1]):
        mb = float(row.rstrip(" [MiB]"))

        if mb > max_free:
            max_idx = i
            max_free = mb

    return max_idx


def is_jean_zay():
    return os.getenv("SLURM_CLUSTER_NAME") == "jean-zay"

def is_oberon_cluster():
    return os.getenv("SLURM_CLUSTER_NAME") == "oberon-hpc"


def rsync_checkpoints_from_jean_zay(pattern):
    if pathlib.Path("/home/nur/").exists():
        home = "/home/nur"
    else:
        home = "/Users/nur"
    cmd = [
        "rsync",
        "-mrvP",
        "-e",
        f"ssh -J nurlan@sapience.dec.ens.fr -i {home}/.ssh/jean_zay",
        f"--include",
        f"*{pattern}*",
        "--exclude",
        "*",
        "uix17ms@jean-zay.idris.fr:/gpfswork/rech/zsq/uix17ms/grnn_data/checkpoints/",
        f"{home}/grnn_data/checkpoints/",
    ]
    print(cmd)
    subprocess.run(cmd)


def rsync_checkpoints_from_oberon(pattern):
    if pathlib.Path("/home/nur/").exists():
        home = "/home/nur"
    else:
        home = str(pathlib.Path.home())

    subprocess.run(
        [
            "rsync",
            "-mrvP",
            "-e",
            f"ssh -J nlan@cognitive-ml.fr -i ~/.ssh/nur-coml",
            "--include",
            f"*{pattern}*",
            "--exclude",
            "*",
            "nlan@oberon:/scratch2/nlan/grnn_data/checkpoints/",
            f"{home}/grnn_data/checkpoints/",
        ]
    )


def get_device():
    if _ENABLE_APPLE_GPU and backends.mps.is_available():
        device = "mps"
    elif cuda.is_available():
        device = f"cuda:{get_free_gpu()}"
    else:
        device = "cpu"
    logging.info(f"Using device {device}")
    return torch.device(device)


def kwargs_to_id(kwargs) -> str:
    s = ""
    for key, val in sorted(kwargs.items()):
        s += f"{key} {val}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
