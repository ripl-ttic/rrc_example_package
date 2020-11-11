#!/usr/bin/env python3
import argparse
import os
import subprocess
import numpy as np

def get_credential():
    with open('user.txt', 'r') as f:
        userid, password = f.read().split()
    return userid, password


def fetch_jobids(cred):
    # submit_result=$(ssh -T -i sshkey ${username}@${hostname} <<<submit)

    # result = subprocess.run(['ssh', '-T', '-i', 'sshkey', username + '@' + hostname, ''], stdout=subprocess.PIPE)
    hostname = 'robots.real-robot-challenge.com'
    username = cred['userid']
    # result = subprocess.run(['ssh', '-v', '-T', '-i', 'sshkey', username + '@' + hostname, '"history"'], stdout=subprocess.PIPE)
    cmd = 'ssh -T -i sshkey ' + username + '@' + hostname + ' <<<history'
    print('fetching job history...')
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode
    print('fetching job history...done')

    lines = out.decode('utf-8').split('\n')
    lines = lines[1:]  # remove the header
    jobids = [str(int(float(line.split()[0]))) for line in lines if line and line.split()[5] != 'X']

    if errcode > 0:
        return None

    return jobids

class Downloader:
    def __init__(self, cred):
        self.username = cred['userid']
        self.password = cred['password']
        self._subprocesses = []

    def download_file(self, filename, logdir, jobid):
        base_url = f'https://robots.real-robot-challenge.com/output/{self.username}/data'
        cmd = f'curl --user {self.username}:{self.password} -o "{logdir}/{jobid}/{filename}" {base_url}/{jobid}/{filename}'
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        out, err = process.communicate()
        errcode = process.returncode
        if errcode > 0:
            return err

    def _download_file_backend(self, filename, logdir, jobid):
        base_url = f'https://robots.real-robot-challenge.com/output/{self.username}/data'
        cmd = f'curl --user {self.username}:{self.password} -o "{logdir}/{jobid}/{filename}" {base_url}/{jobid}/{filename}'
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self._subprocesses.append(process)

    def check_if_exists(self, filename, jobid):
        base_url = f'https://robots.real-robot-challenge.com/output/{self.username}/data'
        cmd = f'curl -sI --user {self.username}:{self.password} -o /dev/null -w "%{{http_code}}" {base_url}/{jobid}/{filename}'
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        out, err = process.communicate()
        errcode = process.returncode
        if errcode > 0:
            print(err)
            return False
        code = int(out.decode('utf-8'))
        return code in [200, 301]

    def download_meta_files(self, logdir, jobid):
        print(f'== downloading meta files (jobid: {jobid}) ==')
        files = ['report.json', 'info.json', 'goal.json', 'user_stdout.txt', 'user_stderr.txt']
        if not os.path.isdir(f'{logdir}/{jobid}'):
            os.mkdir(f'{logdir}/{jobid}')
        for filename in files:
            print(filename)
            error = self.download_file(filename, logdir, jobid)
            if error:
                print(f'download file {filename} failed:\n', error)

        if self.check_if_exists('reward.json', jobid):
            error = self.download_file('reward.json', logdir, jobid)
        else:
            print(f'reward.json does not exist in {jobid}')
        return error

    def download_dat_files(self, logdir, jobid):
        print(f'== downloading dat files (jobid: {jobid}) ==')
        if not os.path.isdir(f'{logdir}/{jobid}'):
            os.mkdir(f'{logdir}/{jobid}')
        files = ['robot_data.dat', 'camera_data.dat']
        for filename in files:
            print(filename)
            error = self.download_file(filename, logdir, jobid)
            if error:
                print(f'download file {filename} failed:\n', error)


def generate_reward_json(logdir, jobid):
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    image = os.path.join(root_dir, 'realrobotchallenge.sif')
    file_dir = os.path.join(root_dir, 'log_manager')

    org_workdir = os.path.join(logdir, jobid)
    workdir = org_workdir
    # '/share/data/' has an issue. If logdir is set there, copy necessary file first and run the program locally.
    if '/share/data' in logdir:
        import shutil
        tmp_workdir = os.path.join('/home/takuma', 'takuma_tmp_workdir')
        print(f'copying {workdir} to {tmp_workdir}')
        if os.path.isdir(tmp_workdir):
            shutil.rmtree(tmp_workdir)
        shutil.copytree(os.path.join(logdir, jobid), tmp_workdir)
        workdir = tmp_workdir

    cmd = f'singularity run --nv {image} python3 {file_dir}/replay_scripts/compute_reward.py {workdir}'
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode

    if errcode > 0:
        print('generate_reward_json failed!:')
        print(err)
        return False
    # copy the generated reward.json back to original dir if using temp work dir
    if org_workdir != workdir:
        print(f'copying back the generated file')
        shutil.copyfile(
            os.path.join(workdir, "reward.json"), os.path.join(org_workdir, "reward.json")
        )


def filter_dir(dirname):
    return int(dirname) <= 9507


def get_filtered_dirs(logdir):
    dirs = os.listdir(logdir)
    dirs = [e for e in dirs if not filter_dir(e)]
    dirs = sorted(list(dirs), key=lambda x: -int(x))
    return dirs


def sanity_check(directory):
    import json
    if not os.path.isfile(os.path.join(directory, 'report.json')):
        return False
    report_file = os.path.join(directory, 'report.json')
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
    except Exception as e:
        print(f'{report_file} is not serializable:')
        print(e)
        print(f'skipping {directory}')
        return False
    if report['backend_error']:
        print(f'backend error is True. skipping {directory}')
        return False
    return True

def get_dirs_that_needs_dat_files(logdir):
    '''
    returns a list of directories that does not contain reward.json
    AND
    does not contain dat files
    '''
    import json
    ret = []
    current_logs = get_filtered_dirs(logdir)
    for directory in current_logs:
        if not sanity_check(os.path.join(logdir, directory)): continue
        dirs = os.listdir(os.path.join(logdir, directory))
        if 'reward.json' not in dirs and ('robot_data.dat' not in dirs or 'camera_data.dat' not in dirs):
            ret.append(directory)
    return ret


def get_dirs_that_need_reward_json(logdir):
    '''
    returns a list of directories that doesn't contain reward.json
    AND
    contain both of robot_data.dat and camera_data.dat
    '''
    import json
    ret = []
    current_logs = get_filtered_dirs(logdir)
    for directory in current_logs:
        if not sanity_check(os.path.join(logdir, directory)): continue
        dirs = os.listdir(os.path.join(logdir, directory))
        if 'reward.json' not in dirs and ('robot_data.dat' in dirs and 'camera_data.dat' in dirs):
            ret.append(directory)
    return ret

commitid2tag = {
    '425760e': 'eval-mpfc+residual+adjust_tip-lvl4',
    '7fe3d16': 'eval-mpfc+residual+adjust_tip-lvl3',
    '77ff1af': 'eval-mpfc+residual+adjust_tip-lvl2',
    '9ffe4f2': 'eval-mpfc+residual+adjust_tip-lvl1',
    'ca89b50': 'eval-mpfc+adjust_tip-lvl4',
    '4640304': 'eval-mpfc+adjust_tip-lvl3',
    '8727243': 'eval-mpfc+adjust_tip-lvl2',
    '0a64666': 'eval-mpfc+adjust_tip-lvl1',
    'a5c186c': 'eval-fc+residual-lvl4',
    '7d3b8bc': 'eval-mpfc+residual-lvl3-corrected',
    '1e74b10': 'eval-mpfc+residual-lvl2-corrected',
    'e049414': 'eval-mpfc+residual-lvl1-corrected',
    'f98b986': 'eval-mpfc+residual-lvl4',
    '6a997c9': 'eval-mpfc+residual-lvl3',
    'f6721ed': 'eval-mpfc+residual-lvl2',
    'e6fb9ab': 'eval-mpfc+residual-lvl1',
    'd6987c3': 'eval-mpfc-lvl1',
    'e8918cd': 'eval-mpfc-lvl2',
    '097616c': 'eval-mpfc-lvl3',
    '11f55cc': 'eval-mpfc-lvl4',
    'ca9f2d5': 'eval-fc-lvl4',
    'e201e3e': 'eval-fc-lvl3',
    '817e7d1': 'eval-fc-lvl2',
    '48583d8': 'eval-fc-lvl1',
}
"""
NOTE: I put the wrong names on some tags when I push it to Github.
eval-mpfc+residual-lvl1 --> eval-fc+residual-lvl1
eval-mpfc+residual-lvl2 --> eval-fc+residual-lvl2
eval-mpfc+residual-lvl3 --> eval-fc+residual-lvl3
"""
correction = {
    "eval-mpfc+residual-lvl1": "eval-fc+residual-lvl1",
    "eval-mpfc+residual-lvl2": "eval-fc+residual-lvl2",
    "eval-mpfc+residual-lvl3": "eval-fc+residual-lvl3",
    "eval-mpfc+residual-lvl3-corrected": "eval-mpfc+residual-lvl3",
    "eval-mpfc+residual-lvl2-corrected": "eval-mpfc+residual-lvl2",
    "eval-mpfc+residual-lvl1-corrected": "eval-mpfc+residual-lvl1",
}
class EvalEpisodes:
    def __init__(self, episodes):
        self.episodes = episodes
        self._preprocess()

    def _preprocess(self):
        self.commitid2episodes = {}
        for episode in self.episodes:
            git_revision = episode['git_revision']
            if git_revision not in self.commitid2episodes:
                self.commitid2episodes[git_revision] = [episode]
            else:
                self.commitid2episodes[git_revision] += [episode]

    def print_stats_table(self, difficulty):
        import pandas as pd
        data = {"tag": [], "avg_reward": [], "stddev": [], "num_episodes": [], "commitid": [] }
        counter = 0
        for commitid, episodes in self.commitid2episodes.items():
            episodes = self.filter_by_difficulty(episodes, difficulty)
            if len(episodes) == 0: continue
            avg_reward, stddev = self.calc_stats(episodes)
            data['commitid'].append(commitid[:7])
            data['avg_reward'].append(avg_reward)
            data['stddev'].append(stddev)
            data['num_episodes'].append(len(episodes))
            if commitid[:7] in commitid2tag:
                tag = commitid2tag[commitid[:7]]
                if tag in correction:
                    tag = correction[tag]
                data['tag'].append(tag)
            else:
                data['tag'].append(None)
            counter += len(episodes)
        df = pd.DataFrame(data)
        df = df.sort_values(by='avg_reward', ascending=False)
        df = df.round({'avg_reward': 2, 'stddev': 2})

        print('============================================================')
        print(f'difficulty: {difficulty}')
        print(df)
        print(f'num total episodes: {counter}')
        print()

    def calc_stats(self, episodes):
        if len(episodes) == 0:
            return None
        rewards = np.asarray([ep['reward'] for ep in episodes])
        return rewards.mean(), np.std(rewards)

    def filter_by_difficulty(self, episodes, difficulty):
        ret = []
        for episode in episodes:
            if episode['difficulty'] == difficulty:
                ret.append(episode)
        return ret


def calculate_reward_stats(logdir):
    import json
    from collections import namedtuple
    current_logs = get_filtered_dirs(logdir)
    episodes = []
    for directory in current_logs:
        if not sanity_check(os.path.join(logdir, directory)): continue

        with open(os.path.join(logdir, directory, 'info.json'), 'r') as f:
            git_revision = json.load(f)['git_revision']
        with open(os.path.join(logdir, directory, 'goal.json'), 'r') as f:
            difficulty = json.load(f)['difficulty']
        with open(os.path.join(logdir, directory, 'reward.json'), 'r') as f:
            reward = json.load(f)['reward']
        episode = {"git_revision": git_revision, "difficulty": difficulty, "reward": reward, 'jobid': int(directory)}
        episodes.append(episode)
    eval_episodes = EvalEpisodes(episodes)
    for difficulty in [1, 2, 3, 4]:
        eval_episodes.print_stats_table(difficulty)


def main(logdir):

    # get list of logs downloaded
    #
    current_logs = get_filtered_dirs(logdir)
    userid, password = get_credential()
    cred = {'userid': userid, 'password': password}

    downloader = Downloader(cred)

    jobids = fetch_jobids(cred)
    logs_to_download = set(jobids).difference(current_logs)
    logs_to_download = [e for e in logs_to_download if not filter_dir(e)]
    logs_to_download = sorted(list(logs_to_download), key=lambda x: -int(x))

    print('current_logs', len(current_logs))
    print('job_ids', len(jobids))
    print('to download', len(logs_to_download))
    # download meta files
    for i, jobid in enumerate(logs_to_download):
        print(f'[Download meta files] progress: {i} / {len(logs_to_download)}')
        downloader.download_meta_files(logdir, jobid)

    # download dat files if a directory doesn't contain reward.json and either of dat files doesn't exist
    target_dirs = get_dirs_that_needs_dat_files(logdir)
    for i, jobid in enumerate(target_dirs):
        print(f'[Download dat files] progress: {i} / {len(target_dirs)}')
        downloader.download_dat_files(logdir, jobid)

    # compute and generate reward.json in the directories that need it
    target_dirs = get_dirs_that_need_reward_json(logdir)
    for i, jobid in enumerate(target_dirs):
        print(f'[Generate reward.json] progress: {i} / {len(target_dirs)}')
        generate_reward_json(logdir, jobid)

    # calculate reward statistsics for every commit id
    calculate_reward_stats(logdir)

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    args = parser.parse_args()

    main(args.logdir)
    # fetch_jobids()
