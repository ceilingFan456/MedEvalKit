# Torch 2.7.0 (CUDA 12.8 wheels)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0

# Optional: matching torchvision / torchaudio
# pip install --index-url https://download.pytorch.org/whl/cu128 torchvision==0.20.0 torchaudio==2.7.0

# Install dependencies
pip install -r requirements.txt
pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation

# For LLaVA-like models
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT && pip install -e .

## download qwen model from huggingface.
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## download show-o model from huggingface.
git clone https://huggingface.co/ShowLab/Show-o

## to install the rest of the packages.
pip install -r pip.txt 

# undo last commit but keep changes staged
git reset --soft HEAD~1