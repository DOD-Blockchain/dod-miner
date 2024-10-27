#!/bin/bash
# Update and upgrade the system
sudo apt update && sudo apt upgrade -y
# Install necessary dependencies
sudo apt-get install -y pkg-config libssl-dev build-essential curl
sudo apt-get install -y pkg-config

sudo apt install nano -y
sudo apt install screen -y
sudo apt install htop -y

# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

# install wsl2 support 
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# libboost
sudo apt-get install libboost-dev
sudo apt install nlohmann-json3-dev

# export dependency
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}'
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
