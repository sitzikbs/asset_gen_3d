#!/bin/bash
# Find the activate script - handle both regular and WSL paths
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f "/home/sitzikbs/VScodeProjects/asset_gen_3d/.venv/bin/activate" ]; then
    source "/home/sitzikbs/VScodeProjects/asset_gen_3d/.venv/bin/activate" 
elif [ -f "/home/sitzikbs/VScodeProjects/asset_gen_3d/.venv/bin/activate" ]; then
    source "/home/sitzikbs/VScodeProjects/asset_gen_3d/.venv/bin/activate"
else
    echo "Could not find virtual environment. Please check if it exists."
    echo "Looked for: .venv/bin/activate"
    echo "Current directory: /home/sitzikbs/VScodeProjects/asset_gen_3d"
    exit 1
fi
