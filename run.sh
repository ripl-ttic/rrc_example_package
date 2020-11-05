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
if (( $# < 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage: $0 <expdir> <cmd>"
fi

expdir=$1

cd $RRC_ROOT
# build dir
if [ ! -d ${expdir}/catkin_ws ]
then
    mkdir -p ${expdir}/catkin_ws/src/usercode
    mkdir -p ${expdir}/logs
    cp -r $RRC_ROOT/python ${expdir}/catkin_ws/src/usercode
    cp -r $RRC_ROOT/*.txt ${expdir}/catkin_ws/src/usercode
    cp -r $RRC_ROOT/*.json ${expdir}/catkin_ws/src/usercode
    cp -r $RRC_ROOT/*.xml ${expdir}/catkin_ws/src/usercode
    cp -r $RRC_ROOT/setup.py ${expdir}/catkin_ws/src/usercode
    cp -r $RRC_ROOT/scripts ${expdir}/catkin_ws/src/usercode
    singularity exec --cleanenv --contain -B ${expdir}/catkin_ws:/ws $RRC_IMAGE bash -c ". /setup.bash; cd /ws; catbuild"
fi
singularity exec --cleanenv --contain --nv -B ${expdir}/catkin_ws:/ws,${expdir}/logs:/logdir,/run,/dev $RRC_IMAGE bash -c \
    ". /setup.bash; . /ws/devel/setup.bash; ${*:2}"
