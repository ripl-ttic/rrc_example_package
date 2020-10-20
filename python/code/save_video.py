#!/usr/bin/env python3
from code.make_env import make_training_env
from code.wrappers import RenderWrapper
from code.utils import set_seed
from gym.wrappers import Monitor
from code.const import TMP_VIDEO_DIR

import statistics
import subprocess
import argparse
import os
import pybullet as p
import time
import shutil

import dl
from dl.rl import set_env_to_eval_mode
import torch
import numpy as np
from code.residual_ppo import ResidualPPO2
dl.rng.seed(0)


def merge_videos(target_path, src_dir):
    src_videos = sorted([os.path.join(src_dir, v) for v in os.listdir(src_dir) if v.endswith('.mp4')])
    command = ['ffmpeg']
    for src_video in src_videos:
        command += ['-i', src_video]
    command += ['-filter_complex', f'concat=n={len(src_videos)}:v=1[outv]', '-map', '[outv]', target_path]
    subprocess.run(command)
    remove_temp_dir(src_dir)


def remove_temp_dir(directory):
    assert directory.startswith('/tmp/'), 'This function can only remove directories under /tmp/'
    shutil.rmtree(directory)


def main(args):
    if args.policy in ['fc', 'mpfc']:
        eval_config = {
            'action_space': 'torque' if args.policy == 'fc' else 'torque_and_position',
            'frameskip': 3,
            'residual': True,
            'reward_fn': f'task{args.difficulty}_competition_reward',
            'termination_fn': 'pos_and_rot_close_to_goal',
            'initializer': f'task{args.difficulty}_eval_init',
            'monitor': True,
            'rank': args.seed,
        }
        env = make_training_env(visualization=True, **eval_config)

    else:
        config = os.path.join(args.exp_dir, 'config.gin')
        bindings = [
            f'make_pybullet_env.reward_fn="task{args.difficulty}_competition_reward"',
            'make_pybullet_env.termination_fn="position_close_to_goal"',
            f'make_pybullet_env.initializer="task{args.difficulty}_eval_init"',
            'make_pybullet_env.visualization=True',
            'make_pybullet_env.monitor=True',
        ]

        dl.load_config(config, bindings)
        ppo = ResidualPPO2(args.exp_dir, nenv=1)
        ppo.load(args.time_steps)
        env = ppo.env
        set_env_to_eval_mode(env)
    acc_rewards = []
    wallclock_times = []
    aligning_steps = []
    env_steps = []
    avg_rewards = []
    for i in range(args.num_episodes):
        start = time.time()
        is_done = False
        observation = env.reset()
        accumulated_reward = 0
        aligning_steps.append(env.unwrapped.step_count)

        #clear some windows in GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #change camera parameters # You can also rotate the camera by CTRL + drag
        p.resetDebugVisualizerCamera( cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

        step = 0
        while not is_done and step < 200:
            step += 1
            if args.policy in ['fc', 'mpfc']:
                action = env.action_space.sample()
                if isinstance(action, dict):
                    action['torque'] *= 0
                    action['position'] *= 0
                else:
                    action *= 0
            else:
                obs = torch.from_numpy(observation).float().to(ppo.device)
                with torch.no_grad():
                    action = ppo.pi(obs, deterministic=True).action.cpu().numpy()[0]
            observation, reward, is_done, info = env.step(action)
            accumulated_reward += reward
        acc_rewards.append(accumulated_reward)
        env_steps.append(env.unwrapped.step_count - aligning_steps[-1])
        avg_rewards.append(accumulated_reward / env_steps[-1])
        print("Episode {}\tAccumulated reward: {}".format(i, accumulated_reward))
        print("Episode {}\tAlinging steps: {}".format(i, aligning_steps[-1]))
        print("Episode {}\tEnv steps: {}".format(i, env_steps[-1]))
        print("Episode {}\tAvg reward: {}".format(i, avg_rewards[-1]))
        end = time.time()
        print('Elapsed:', end - start)
        wallclock_times.append(end - start)

    env.close()

    def _print_stats(name, data):
        print('======================================')
        print(f'Mean   {name}\t{np.mean(data):.2f}')
        print(f'Max    {name}\t{max(data):.2f}')
        print(f'Min    {name}\t{min(data):.2f}')
        print(f'Median {name}\t{statistics.median(data):2f}')

    print('Total elapsed time\t{:.2f}'.format(sum(wallclock_times)))
    print('Mean elapsed time\t{:.2f}'.format(sum(wallclock_times) / len(wallclock_times)))
    _print_stats('acc reward', acc_rewards)
    _print_stats('aligning steps', aligning_steps)
    _print_stats('step reward', avg_rewards)

    if args.time_steps is None:
        args.time_steps = ppo.ckptr.ckpts()[-1]
    if args.filename is None:
        video_file = os.path.join(args.exp_dir, "model_{}_steps_level_{}.mp4".format(args.time_steps, args.difficulty))
    else:
        video_file = os.path.join(args.exp_dir, "{}_{}_steps_level_{}.mp4".format(args.filename, args.time_steps, args.difficulty))
    merge_videos(video_file, TMP_VIDEO_DIR)
    print('Video is saved at {}'.format(video_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", help="experiment_dir")
    parser.add_argument("--time_steps", type=int, default=None, help="time steps")
    parser.add_argument("--difficulty", type=int, default=1, help="difficulty")
    parser.add_argument("--num_episodes", default=10, type=int, help="number of episodes to record a video")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--policy", default='ppo', choices=["fc", "mpfc", "ppo"], help="which policy to run")
    parser.add_argument("--filename", help="filename")
    args = parser.parse_args()

    print('For faster recording, run `git apply faster_recording_patch.diff`. This temporary changes episode length and window size.')

    # check if `ffmpeg` is available
    if shutil.which('ffmpeg') is None:
        raise OSError('ffmpeg is not available. To record a video, you need to install ffmpeg.')

    if os.path.isdir(TMP_VIDEO_DIR):
        remove_temp_dir(TMP_VIDEO_DIR)
    os.mkdir(TMP_VIDEO_DIR)

    if args.policy in ['fc', 'mpfc']:
        scripted_video_dir = 'rrc_videos'
        args.time_steps = 0
        if not os.path.isdir(scripted_video_dir):
            os.makedirs(scripted_video_dir)
        args.exp_dir = scripted_video_dir
    else:
        assert args.exp_dir is not None

    set_seed(args.seed)
    # gym.logger.set_level(gym.logger.INFO)
    main(args)
