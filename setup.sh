#!/bin/bash
# Use python 3.12 with pyenv
pyenv install 3.12
pyenv local 3.12
python --version

# Check if a venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

which python

pip install --upgrade pip setuptools wheel

# Install Dependencies
pip install -r requirements.txt

# Optional dependency used for USDZ->OBJ conversion in view_reconstruction.py.
# Some environments may not have a compatible aspose-3d build.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Skipping aspose-3d on macOS arm64 (no compatible wheel currently available)."
    echo "If you need USDZ conversion, use an alternative converter or an x86_64 Python environment."
elif ! pip install aspose-3d; then
    echo "Warning: Could not install aspose-3d in this environment."
    echo "If you need USDZ conversion, install it manually in a compatible environment."
fi