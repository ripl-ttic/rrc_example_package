#!/usr/bin/env python3
"""Execute a submission"""
import argparse
import os
import subprocess
import shutil
from shutil import ignore_patterns
import yaml

parser = argparse.ArgumentParser(description='Train Agent.')
parser.add_argument('config', type=str, help='config')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f)

log_dir = os.path.join(config['logdir'], 'logs')
ws_dir = os.path.join(config['logdir'], 'catkin_ws')
code_dir = os.getenv("RRC_ROOT")
if code_dir is None:
    raise ValueError("RRC_ROOT env variable not set.")

image_path = os.getenv("RRC_IMAGE")
if code_dir is None:
    raise ValueError("RRC_IMAGE env variable not set.")

if not os.path.exists(ws_dir):
    os.makedirs(os.path.join(ws_dir, 'src'))
    shutil.copytree(code_dir, os.path.join(ws_dir, 'src', 'usercode'),
                    ignore=ignore_patterns("log*", "*.sif"),
                    ignore_dangling_symlinks=True)

    build_cmd = [
        "singularity",
        "exec",
        "--cleanenv",
        "--contain",
        "-B",
        "{}:/ws".format(ws_dir),
        image_path,
        "bash",
        "-c",
        ". /setup.bash; cd /ws; catbuild",
    ]
    proc = subprocess.run(
        build_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # copy config file into singularity image
    os.makedirs(log_dir)
    config['logdir'] = '/logdir'
    shutil.copyfile(config['base_config'], os.path.join(log_dir, 'base.gin'))
    config['base_config'] = '/logdir/base.gin'
    with open(os.path.join(log_dir, 'singularity_config.yaml'), 'w') as f:
        yaml.dump(config, f)


# print singularity mount points
print(ws_dir + ':/ws,' + log_dir + ':/logdir')
