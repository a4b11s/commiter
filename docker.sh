#! /bin/bash

docker rm commiter ; docker image rm "a4b11s/commiter"    \
 ; docker build --tag "a4b11s/commiter" . \
 && docker run --memory=7g --memory-swap=-1 --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v ./data:/data --name "commiter" "a4b11s/commiter"
