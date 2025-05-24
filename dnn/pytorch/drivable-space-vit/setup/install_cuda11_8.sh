wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get -y install cudnn-cuda-11

