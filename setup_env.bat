@echo off
REM Create and activate the conda environment for asset_gen
conda env create -f environment.yml
conda activate asset_gen
REM (Optional) Print Python and CUDA info
echo Python version:
python --version
echo CUDA available:
python -c "import torch; print(torch.cuda.is_available())"
