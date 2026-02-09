#!/bin/bash
# Run the worker_002 re-run + merge script inside awe_venv
# Usage:
#   cd /workspace/awe/example
#   bash run_rerun_and_merge.sh

set -e

echo "Activating awe_venv..."
eval "$(conda shell.bash hook)"
conda activate awe_venv

echo "Python: $(which python)"
echo "Working directory: $(pwd)"

cd /workspace/awe/example
python rerun_worker002_and_merge.py
