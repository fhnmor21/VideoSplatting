#!/bin/sh
podman run -it \
    --cap-add=SYS_PTRACE \
    --privileged=true \
    --shm-size=48GB \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $HOME:$HOME \
    --name rocm_pytorch \
    docker.io/rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
    #--ipc=host \

