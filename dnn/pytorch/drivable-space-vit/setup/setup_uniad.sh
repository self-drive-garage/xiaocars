conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0


conda install -c omgarcia gcc-6 # gcc-6.2

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118