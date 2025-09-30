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