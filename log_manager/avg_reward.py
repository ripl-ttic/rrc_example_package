import os
import json

job_low = 9497
job_high = 9507

rews = []
for job in range(job_low, job_high):
    rew_file = f'logs/{job}/reward.txt' 
    if os.path.exists(rew_file):
        os.rename(rew_file, rew_file[:-3] + 'json')
    rew_file = f'logs/{job}/reward.json' 
    if os.path.exists(rew_file):
        with open(rew_file, 'r') as f:
            rew = json.load(f)['reward']
            rews.append(rew)


print(len(rews))
print(sum(rews) / len(rews))
