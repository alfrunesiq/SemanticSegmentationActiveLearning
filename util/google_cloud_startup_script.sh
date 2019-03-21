#!/bin/bash

apt update
apt upgrade -y
apt install -y git python3-pip

# Add contrib and experimental repositories
sed -i.bak 's/stretch[^ ]* main$/& contrib non-free/g' /etc/apt/sources.list
echo "deb http://httpredir.debian.org/debian experimental main contrib non-free" >> /etc/apt/sources.list
echo "deb http://httpredir.debian.org/debian buster main contrib non-free" >> /etc/apt/sources.list
apt update

# Install nvidia proprietary stuff
apt install -y -t experimental libc-bin
apt install -y -t experimental nvidia-driver
apt install -y -t experimental nvidia-cuda-dev

# Get cudnn
wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-8.0-linux-x64-v7.tgz
tar -xzvf cudnn-8.0-linux-x64-v7.tgz
cp cuda/lib64/* /usr/lib/
cp cuda/includa/* /usr/include/

# Setup python3 with Tensorflow
pip3 install tensorflow-gpu
pip3 install tqdm

# Execute and run code here (Mount external disk?)
mkdir tensorflow
mount /dev/sdb1 tensorflow

python3 -c "import tensorflow; import tqdm"

shutdown now
