#!/bin/bash
if (( $# != 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <log directory> <singularity image>"
    exit 1
fi
logdir=$1
image=$2

hostname=robots.real-robot-challenge.com
username=`cat user.txt | head -n 1`

jobs=$(ssh -T -i sshkey ${username}@${hostname} <<< history | tail -n +2 | awk -F'[. ]' '{print $1, $14}')

echo ${jobs} | xargs -n 2 echo
echo ${jobs} | xargs -n 2 -P 10 bash after_job.sh ${logdir} ${image}
