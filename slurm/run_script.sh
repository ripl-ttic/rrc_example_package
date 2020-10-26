#! /bin/bash
if [ -z "$RRC_ROOT" ]
then
      echo "Please set \$RRC_ROOT (e.g., export RRC_ROOT=/path/to/your/rrc_phase_2)"
      exit 1
fi

if [ -z "$RRC_IMAGE" ]
then
      echo "Please set \$RRC_IMAGE to the path of your singularity image"
      exit 1
fi
cd $RRC_ROOT
mounts=`python slurm/build_ws.py $@`
echo ${mounts}
# singularity exec --cleanenv --contain --nv -B ${mounts} $RRC_IMAGE bash -c \
#     ". /setup.bash; . /ws/devel/setup.bash; python3 /ws/src/usercode/python/code/train_ppo.py /logdir/singularity_config.yaml"
singularity exec --cleanenv --contain --nv -B ${mounts} $RRC_IMAGE bash -c \
    ". /setup.bash; . /ws/devel/setup.bash; bash"
