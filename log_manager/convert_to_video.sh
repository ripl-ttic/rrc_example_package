#!/usr/bin/env bash
set -x

logdir=$1
jobid=$2
server_logdir="log-${jobid}"
echo $server_logdir
# scp -r ${logdir} takumasus:workspace/rrc_phase_2/log_manager/${server_logdir}
ssh -t takumasus "cd ~/workspace/rrc_phase_2/log_manager && ./make_video.sh ~/Downloads/rrc_phase_2.sif ${server_logdir}"
scp takumasus:workspace/rrc_phase_2/log_manager/${server_logdir}/video.mp4 ${logdir}/video.mp4
scp takumasus:workspace/rrc_phase_2/log_manager/${server_logdir}/comparison.mp4 ${logdir}/comparison.mp4
