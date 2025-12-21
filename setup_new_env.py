"""
setup_new_env.py
----------------
Automatic Conda environment setup script for the IDS thesis.
OPTIMIZED FOR RTX 5060 Ti: Automatically installs PyTorch with CUDA 12.8
(Blackwell sm_120 support).

Usage:
    python setup_new_env.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ENV_NAME = "ids-thesis"
PYTHON_VERSION = "3.11"   # Python 3.11 offers best compatibility with CUDA 12.8+
PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_DIR / "requirements.txt"

# PyTorch download link for RTX 50-series (CUDA 12.8 Stable)
# If you want to test CUDA 13.0 Nightly:
#     https://download.pytorch.org/whl/nightly/cu130
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def run(cmd: str, allow_fail: bool = False) -> None:
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        if not allow_fail:
            print(f"[!] Error: {cmd}")
            sys.exit(result.returncode)
        else:
            print(f"[i] Command failed but allowed to continue.")

def ensure_conda_available():
    if shutil.which("conda") is None:
        print("\n[!] Could not find the `conda` command. Please run this inside Anaconda Prompt.")
        sys.exit(1)

def conda_run(cmd_in_env: str) -> None:
    """Run a command inside the newly created environment."""
    run(f"conda run -n {ENV_NAME} {cmd_in_env}")

# ============================================================
# 3. SETUP PROCESS
# ============================================================

if __name__ == "__main__":
    print(f"--- SETTING UP ENVIRONMENT FOR RTX 5060 Ti ({ENV_NAME}) ---")
    ensure_conda_available()

    if not REQUIREMENTS_PATH.exists():
        print(f"[!] requirements.txt not found: {REQUIREMENTS_PATH}")
        sys.exit(1)

    # 1. Remove old env (ensure clean installation for CUDA)
    print(f"\n[1] Removing old environment '{ENV_NAME}'...")
    run(f"conda env remove -n {ENV_NAME} -y", allow_fail=True)

    # 2. Create new environment
    print(f"\n[2] Creating new environment (Python {PYTHON_VERSION})...")
    run(f"conda create -n {ENV_NAME} python={PYTHON_VERSION} -y")

    # 3. Install PyTorch CUDA 12.8 (MOST IMPORTANT)
    print(f"\n[3] Installing PyTorch with CUDA 12.8 for RTX 5060 Ti...")
    conda_run("python -m pip install --upgrade pip")
    
    # Install Torch first to avoid dependency conflicts
    conda_run(f"pip install torch torchvision torchaudio --index-url {TORCH_INDEX_URL}")

    # 4. Install supporting libraries
    print("\n[4] Installing supporting packages (Jupyter, Pandas, mRMR, etc.)...")
    conda_run("pip install jupyterlab notebook ipykernel")
    conda_run(f"pip install -r {REQUIREMENTS_PATH}")

    # 5. Register Jupyter kernel
    print("\n[5] Registering Jupyter Kernel...")
    display_name = f"{ENV_NAME} (RTX 5060Ti)"
    conda_run(
        f'python -m ipykernel install --user --name "{ENV_NAME}" --display-name "{display_name}"'
    )

    # 6. GPU verification
    print("\n[6] Verifying GPU functionality with PyTorch...")
    check_gpu_code = (
        "import torch; "
        "print('Torch Version:', torch.__version__); "
        "print('CUDA Available:', torch.cuda.is_available()); "
        "print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    )
    conda_run(f'python -c "{check_gpu_code}"')

    print("\n✅ SETUP COMPLETE! GPU IS READY.")
    print(f"👉 Activate with:  conda activate {ENV_NAME}")
