import os
from datetime import datetime
from pathlib import Path
from . import config

def setup_experiment_folder():
    """
    Tạo cấu trúc thư mục cho experiment mới dựa trên thời gian hiện tại.
    Structure:
        results/experiments/exp_YYYYMMDD_HHMMSS/
            ├── models/          (Chứa scaler, ae, rf)
            ├── figures/         (Chứa confusion matrix)
            └── report.txt       (Kết quả metric)
    """
    # 1. Tạo ID dựa trên thời gian
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    
    # 2. Định nghĩa đường dẫn
    exp_dir = config.RESULTS_DIR / "experiments" / exp_name
    models_dir = exp_dir / "models"
    figures_dir = exp_dir / "figures"
    
    # 3. Tạo thư mục
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 New Experiment Created: {exp_dir}")
    
    return {
        "root": exp_dir,
        "models": models_dir,
        "figures": figures_dir
    }


def log_experiment_details(exp_path=None):
    """
    In và lưu thông tin chi tiết về cấu hình Experiment.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = [
        "==================================================",
        f"       EXPERIMENT SUMMARY - {now}",
        "==================================================",
        f"🚀 Device: {config.DEVICE}", #
        f"📂 Exp Path: {exp_path if exp_path else 'Not Specified'}",
        "",
        "--- DATA CONFIGURATION ---",
        f"🔹 Binary Mode: {config.BINARY_MODE}", #
        f"🔹 Random Seed: {config.SEED}", #
        f"🔹 Selected Features (Total): {len(config.SELECTED_FEATURES)}", #
        f"🔹 mRMR Features (K): {len(config.mRMR_FEATURES)}", #
        "",
        "--- AUTOENCODER HYPERPARAMETERS ---",
        f"🔸 Input Dim: {config.AE_INPUT_DIM}", #
        f"🔸 Latent Dim (N): {config.AE_LATENT_DIM}", #
        f"🔸 Hidden Layers: {config.AE_HIDDEN_LAYERS}", #
        f"🔸 Epochs: {config.AE_EPOCHS}", #
        f"🔸 Batch Size: {config.AE_BATCH_SIZE}", #
        f"🔸 Learning Rate: {config.AE_LR}", #
        "",
        "--- RANDOM FOREST HYPERPARAMETERS ---",
        f"🔸 Estimators: {config.RF_ESTIMATORS}", #
        f"🔸 Max Depth: {config.RF_MAX_DEPTH}", #
        "=================================================="
    ]
    
    # In ra console
    content = "\n".join(summary)
    print(content)
    
    # Lưu vào file txt trong folder experiment (nếu có path)
    if exp_path:
        log_file = exp_path / "experiment_details.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Experiment details saved to: {log_file}")