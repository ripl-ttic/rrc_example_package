#!/bin/bash
if (( $# != 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <log directory> <singularity image>"
    exit 1
fi
logdir=$1
image=$2
dir=`dirname $0`

if [ -z "$RRC_ROOT" ]
then
      echo "Please set \$RRC_ROOT (e.g., export RRC_ROOT=/path/to/your/rrc_phase_2)"
      exit 1
fi

hostname=robots.real-robot-challenge.com
username=`cat ${dir}/user.txt | head -n 1`

jobs=$(ssh -o "StrictHostKeyChecking no" -T -i ${dir}/sshkey ${username}@${hostname} <<< history | tail -n +2 | tr -s " " | awk -F'[. ]' '{print $1, $7}')

echo ${jobs} | xargs -n 2 echo
echo ${jobs} | xargs -t -n 2 -P 1 bash ${dir}/launch_slurm_job.sh ${logdir} ${image}
