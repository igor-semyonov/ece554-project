#!/bin/bash
#sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-tensorrt:r8.0.1-runtime
CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"0"}
DOCKER_BRIDGE=${3:-"host"}
#restart a stopped container with output
#sudo docker start -ai <container name>
sudo docker run -it --net=$DOCKER_BRIDGE --gpus all --shm-size=16g -v $(dirname $(pwd)):/root/space/projects --runtime nvidia -e DISPLAY=$DISPLAY mypytorch_jet /bin/bash
#--gpus device=$NV_VISIBLE_DEVICES \
#--net=$DOCKER_BRIDGE \
# --shm-size=16g \
# -v $(dirname $(pwd)):/root/space/projects \
# $CMD
