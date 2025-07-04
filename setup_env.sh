#!/bin/bash
# 3D Asset Generation Environment Setup Script
# This script creates a Python virtual environment and installs all required dependencies

# Exit on error
set -e

# Configuration
VENV_NAME=".venv"
PYTHON_VERSION="3.10"
CUDA_VERSION="121"  # For CUDA 12.1

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BOLD}${GREEN}=== 3D Asset Generation Environment Setup ===${NC}"

# Check for Python
echo -e "\n${BOLD}Checking for Python...${NC}"
if command -v python3.10 &>/dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Python not found. Please install Python ${PYTHON_VERSION} first.${NC}"
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
echo -e "Found Python $PY_VERSION"

# Check for CUDA
echo -e "\n${BOLD}Checking for CUDA...${NC}"
if command -v nvcc &>/dev/null; then
    CUDA_INSTALLED=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c 1-4)
    echo -e "Found CUDA $CUDA_INSTALLED"
else
    echo -e "${YELLOW}CUDA not found. GPU acceleration may not be available.${NC}"
fi

# Create and activate virtual environment
echo -e "\n${BOLD}Creating virtual environment...${NC}"
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf "$VENV_NAME"
fi

$PYTHON_CMD -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "\n${BOLD}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch dependencies first
echo -e "\n${BOLD}Installing PyTorch with CUDA ${CUDA_VERSION}...${NC}"
pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Install remaining packages
echo -e "\n${BOLD}Installing dependencies...${NC}"
pip install diffusers==0.34.0 transformers==4.53.2 huggingface_hub==0.33.1
pip install sentencepiece accelerate==1.8.1 tokenizers==0.21.2
pip install charset_normalizer chardet protobuf pillow numpy
pip install "xformers==0.0.27.post2" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# Install bitsandbytes with correct CUDA version
echo -e "\n${BOLD}Installing bitsandbytes for CUDA ${CUDA_VERSION}...${NC}"
# export BNB_CUDA_VERSION=$CUDA_VERSION
pip install bitsandbytes==0.46.1

# Verify installations
echo -e "\n${BOLD}Verifying installations...${NC}"
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"
$PYTHON_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
$PYTHON_CMD -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
$PYTHON_CMD -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"
$PYTHON_CMD -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Set environment variables to a file for future sessions
echo -e "\n${BOLD}Creating activation helpers...${NC}"
cat > activate_env.sh << EOF
#!/bin/bash
# Find the activate script - handle both regular and WSL paths
if [ -f "$VENV_NAME/bin/activate" ]; then
    source "$VENV_NAME/bin/activate"
elif [ -f "$(pwd)/$VENV_NAME/bin/activate" ]; then
    source "$(pwd)/$VENV_NAME/bin/activate" 
elif [ -f "$PWD/$VENV_NAME/bin/activate" ]; then
    source "$PWD/$VENV_NAME/bin/activate"
else
    echo "Could not find virtual environment. Please check if it exists."
    echo "Looked for: $VENV_NAME/bin/activate"
    echo "Current directory: $(pwd)"
    exit 1
fi

export BNB_CUDA_VERSION=$CUDA_VERSION
echo "Environment activated with CUDA $CUDA_VERSION"
EOF
chmod +x activate_env.sh

echo -e "\n${BOLD}${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To activate this environment in the future, run:${NC}"
echo -e "    ${BOLD}source ./activate_env.sh${NC}"
