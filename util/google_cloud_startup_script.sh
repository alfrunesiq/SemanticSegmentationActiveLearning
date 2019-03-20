#!/bin/bash

apt update
apt upgrade -y
apt install -y git python3-pip

sed -i.bak 's/stretch[^ ]* main$/& contrib non-free/g' /etc/apt/sources.list
echo "deb http://httpredir.debian.org/debian experimental main contrib non-free" >> /etc/apt/sources.list
echo "deb http://httpredir.debian.org/debian buster main contrib non-free" >> /etc/apt/sources.list
apt update
apt install -y linux-headers-$(uname -r)
apt install -y -t experimental nvidia-driver nvidia-cuda-dev

# Setup python3 with Tensorflow
pip3 install tensorflow-gpu
pip3 install tqdm

# Execute and run code here (Mount external disk?)

# Notify, wait a minute and shut down
