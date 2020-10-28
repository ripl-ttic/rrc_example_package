#!/usr/bin/env python3
"""Execute a submission"""
import argparse
import os
import signal
import subprocess
import tempfile
import time
import json
import traceback
import socket
import logging
import pathlib
import sys
import shutil
from run_in_simulation import LocalExecutionConfig, SubmissionRunner


def main():
    log_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format="[SUBMISSION %(levelname)s %(asctime)s] %(message)s",
        level=logging.DEBUG,
        handlers=[log_handler],
    )

    config = LocalExecutionConfig()
    runner = SubmissionRunner(config)

    logdir = '/output'
    replay_file = '/output/comparison.avi'
    # command = 'rosrun rrc replay.py ' + custom_logfile + ' ' + replay_file
    command = 'python3 /ws/src/usercode/log_manager/replay_scripts/replay.py ' + logdir + ' ' + replay_file
    try:
        with tempfile.TemporaryDirectory(
            prefix="run_submission-"
        ) as ws_dir:
            logging.info("Use temporary workspace %s", ws_dir)

            user_returncode = None

            # create "src" directory and cd into it
            src_dir = os.path.join(ws_dir, "src")
            os.mkdir(src_dir)
            os.chdir(src_dir)

            runner.clone_user_repository()
            # runner.load_goal(src_dir)
            # runner.store_info()
            os.chdir(ws_dir)
            runner.build_workspace(ws_dir)

            user_returncode = runner.run_user_command(ws_dir, command)

            runner.store_report(False, user_returncode)

            logging.info("Finished.")
    except Exception as e:
        logging.critical("FAILURE: %s", e)

        # FIXME just for debugging, remove later
        traceback.print_exc()

        error_report_file = os.path.join(
            runner.config.host_output_dir, "error_report.txt"
        )
        with open(error_report_file, "w") as fh:
            fh.write(
                "Submission failed with the following error:\n{}\n".format(
                    e
                )
            )


if __name__ == "__main__":
    main()
