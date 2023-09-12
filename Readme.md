# env in container
rm /etc/apt/sources.list.d/cuda.list
rm /etc/apt/sources.list.d/nvidia-ml.list

apt-get update
apt-get upgrade -y

wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
chmod +x Anaconda3-2023.03-Linux-x86_64.sh
./Anaconda3-2023.03-Linux-x86_64.sh
rm Anaconda3-2023.03-Linux-x86_64.sh

source ~/.bashrc
conda create --name ml python=3.10
conda activate ml

#161
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

#163
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install wandb mlxtend opencv-python imutils tqdm