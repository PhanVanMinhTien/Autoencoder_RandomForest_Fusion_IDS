# IDS AE+RF Fusion - Network Intrusion Detection System

---

## Table of Contents
- [English Version](#english-version)
- [Phiên Bản Tiếng Việt](#phiên-bản-tiếng-việt)

---

# English Version

## Project Description

This project develops a **Network Intrusion Detection System (IDS)** using a **hybrid approach combining Autoencoder (AE) and Random Forest (RF)** to detect and classify network attacks.

### Main Objectives:
- **Dimensionality Reduction**: Use Autoencoder to extract potent features from network traffic data
- **Classification**: Use Random Forest to classify data into "Benign" or "Attack" categories
- **Method Comparison**: Evaluate AE+RF performance against baseline methods (RF-only, SVM)
- **Cross-Dataset Evaluation**: Assess generalization capability across different datasets

---

## Project Structure

```
ids_ae_rf_fusion/
├── README.md                           # This documentation
├── requirements.txt                    # Python library dependencies
├── setup_new_env.py                    # Environment initialization script
│
├── src/                                # Main source code
│   ├── config.py                       # Project configuration (paths, hyperparameters)
│   ├── autoencoder.py                  # Deep Autoencoder Model (PyTorch)
│   ├── rf_classifier.py                # Random Forest Classifier
│   ├── preprocessing.py                # Data preprocessing and cleaning
│   ├── feature_selection.py            # Feature selection (mRMR)
│   ├── evaluation.py                   # Model evaluation & visualization
│   └── utils.py                        # Utility functions
│
├── datasets/                           # Datasets (metadata only)
│   └── dataset.txt                     # Dataset description
│
├── notebooks/                          # Jupyter Notebooks (experimental pipeline)
│   ├── 0a_mRMR_selection.ipynb         # Feature selection (mRMR)
│   ├── 0b_mRMR_features_and_latent_features.ipynb
│   │
│   ├── 1a_ae_rf_fusion_mix_training.ipynb      # [Stage 1] Mixed training
│   ├── 1b_rf_mix_training.ipynb                # Baseline: RF-only
│   ├── 1c_svm_mix_training.ipynb               # Baseline: SVM
│   │
│   ├── 2a_ae_rf_fusion_within_dataset.ipynb    # [Stage 2] Within-dataset testing
│   ├── 2b_rf_within_dataset.ipynb
│   ├── 2c_svm_within_dataset.ipynb
│   │
│   ├── 3a_ae_rf_fusion_cross_dataset.ipynb     # [Stage 3] Cross-dataset testing
│   ├── 3b_rf_cross_dataset.ipynb
│   ├── 3c_svm_cross_dataset.ipynb
│   │
│   └── archived/                       # Legacy notebooks & experiments
│
├── results/                            # Experimental results
│   ├── experiments/                    # Results from each run
│   │   └── exp_YYYYMMDD_HHMMSS/
│   │       ├── report_*.txt            # Detailed metrics report
│   │       ├── figures/                # Confusion Matrix & visualizations
│   │       ├── models/                 # Trained models
│   │       └── experiment_details.txt  # Configuration & hyperparameters
│   │
│   └── Summary/                        # Comparison summary of methods
│
└── models/                             # (Optional) Trained model storage
```

---

## Requirements & Installation

### 1. **System Requirements**
- Python 3.8+
- CUDA (optional, for GPU acceleration if available)
- RAM: 8GB+ (16GB+ recommended)

### 2. **Library Installation**

```bash
# Method 1: Direct installation from requirements.txt
pip install -r requirements.txt

# Method 2: Using setup script (if available)
python setup_new_env.py
```

### 3. **Main Libraries**

Core libraries used:
- **numpy**, **pandas** - Data processing
- **scikit-learn** - Machine Learning (Random Forest, SVM, metrics)
- **torch** - Deep Learning (Autoencoder)
- **matplotlib**, **seaborn** - Result visualization
- **joblib** - Model saving/loading
- **mrmr-selection** - Feature selection

---

## Datasets

The project uses two public IDS datasets:

| Dataset | Year | Samples | Characteristics |
|---------|------|---------|-----------------|
| **CIC-IDS2017** | 2017 | ~2.8M | Real network traffic, 15 attack types |
| **CSE-CIC-IDS2018** | 2018 | ~2.5M | Updated from 2017, modern attacks |

### Attack Types:
- **BENIGN** - Normal traffic
- **DOS** - Denial of Service (HULK, GoldenEye, SlowLoris, SlowHTTPTest)
- **DDOS** - Distributed DoS (HOIC, LOIC)
- **BRUTEFORCE** - Brute force attacks (FTP, SSH)
- **BOT** - Botnet
- **PORTSCAN** - Port scanning
- **WEB** - Web attacks (SQL Injection, XSS, Brute Force)
- **INFILTRATION** - Infiltration
- **HEARTBLEED** - Heartbleed vulnerability

---

## Main Module Descriptions

### 1. **config.py** - Project Configuration

Manages all configurations:
- **Data paths**: Datasets, results directories
- **Data hyperparameters**:
  - `BINARY_MODE = True` - Binary classification (Benign vs Attack)
  - `CHUNK_SIZE = 100000` - File chunk reading size
  - `SEED = 42` - Random seed for reproducibility

- **Preprocessing**:
  - `DROP_COLS` - Columns to drop (identifiers, sparse, mismatches)
  - `RENAME_2018_TO_2017` - Column name mapping 2018 → 2017 (normalization)
  - `BENIGN_LABELS` - Labels considered as "benign"

- **Autoencoder hyperparameters**:
  - Input dimension, latent dimension, hidden layers

- **Random Forest hyperparameters**:
  - Number of trees (n_estimators)
  - Max depth
  - Class weight balancing

### 2. **autoencoder.py** - Deep Autoencoder

```python
class DeepAutoencoder(nn.Module):
    """
    Symmetric autoencoder with:
    - Encoder: Input → Hidden Layers → Latent (bottleneck)
    - Decoder: Latent → Hidden Layers → Output
    
    Features:
    - Batch Normalization + LeakyReLU
    - Linear output activation
    - MSE Loss for data reconstruction
    """
```

**Functions**:
- Reconstruction: Learn to extract features from normal data
- Dimensionality Reduction: Compress 67 features to smaller latent space
- Anomaly Detection: Attacks have high reconstruction error

### 3. **rf_classifier.py** - Random Forest Classifier

```python
def train_rf(X_train, y_train, save_path=None):
    """
    Train Random Forest with:
    - n_estimators = 200
    - max_depth = 20
    - class_weight = 'balanced' (handles data imbalance)
    - n_jobs = 8 (multi-processing)
    """
```

### 4. **preprocessing.py** - Data Preprocessing

- Read data in chunks
- Data cleaning (remove NaN, duplicates)
- Column name normalization (2018 → 2017)
- Drop identifier columns (Flow ID, IP, Timestamp)
- Data standardization (StandardScaler)
- Label encoding (Benign → 0, Attack → 1)

### 5. **feature_selection.py** - Feature Selection

Uses **mRMR (Minimum Redundancy Maximum Relevance)**:
- Select features with high correlation to labels
- Prioritize features with low redundancy

### 6. **evaluation.py** - Model Evaluation

Computed metrics:
- **Accuracy** - Overall accuracy
- **MCC (Matthews Correlation Coefficient)** - Balanced metric for imbalanced data
- **Precision, Recall, F1-score** - Per-class metrics
- **Confusion Matrix** - True/false positives visualization

---

## Usage Guide

### **Stage 0: Data Preparation**
1. Download CIC-IDS2017 & CSE-CIC-IDS2018 to `datasets/CIC-IDS2017` and `datasets/CSE-CIC-IDS2018`
2. Run notebook `0a_mRMR_selection.ipynb` for feature selection
3. Result: List of ~20-30 best features

### **Stage 1: Mixed Training**
- Combine data from both 2017 and 2018
- Train: 80% / Test: 20%
- Run:
  - `1a_ae_rf_fusion_mix_training.ipynb` - **Proposed method**
  - `1b_rf_mix_training.ipynb` - RF-only baseline
  - `1c_svm_mix_training.ipynb` - SVM baseline
- Result: Compare 3 methods

### **Stage 2: Within-Dataset Testing**
- Train & test on same dataset:
  - 2017 train/test
  - 2018 train/test
- Run:
  - `2a_ae_rf_fusion_within_dataset.ipynb`
  - `2b_rf_within_dataset.ipynb`
  - `2c_svm_within_dataset.ipynb`

### **Stage 3: Cross-Dataset Testing**
- Train on one dataset, test on another:
  - Train 2017 → Test 2018
  - Train 2018 → Test 2017
- Run:
  - `3a_ae_rf_fusion_cross_dataset.ipynb`
  - `3b_rf_cross_dataset.ipynb`
  - `3c_svm_cross_dataset.ipynb`
- **Purpose**: Evaluate generalization capability

---

## Expected Results

### Performance Expectations
| Method | Mixed | Within-2017 | Within-2018 | Cross (17→18) | Cross (18→17) |
|--------|-------|-------------|-------------|---------------|---------------|
| **AE+RF (Proposed)** | ~95-98% | ~97-99% | ~95-97% | ~85-90% | ~82-87% |
| **RF-only (Baseline)** | ~92-95% | ~95-98% | ~93-96% | ~80-85% | ~78-83% |
| **SVM (Baseline)** | ~90-94% | ~93-97% | ~91-95% | ~75-82% | ~73-80% |

**Note**: Cross-dataset results typically lower due to distribution differences between datasets

---

## How to Run a Notebook

### 1. **From Command Line**
```bash
# Run notebook and save results
jupyter nbconvert --to notebook --execute --output output.ipynb 1a_ae_rf_fusion_mix_training.ipynb

# Or open Jupyter Lab for interactive execution
jupyter lab
```

### 2. **In VS Code**
- Open `.ipynb` file
- Select Python kernel
- Run individual cells or Run All

### 3. **Change Configuration**
- Edit `src/config.py`:
  - Data paths
  - Model hyperparameters
  - Feature selection parameters
- Configuration automatically loaded in notebooks

---

## Results Structure

Each notebook run creates a results directory:
```
results/experiments/exp_YYYYMMDD_HHMMSS/
├── experiment_details.txt          # Configuration, hyperparameters
├── report_Mixed_Test_Set.txt       # Detailed metrics report
├── figures/
│   ├── cm_Mixed_Test_Set.png       # Confusion Matrix
│   └── ... (other figures)
└── models/
    └── model.joblib / model.pt     # Trained models
```

---

## Key Points

**Advantages of AE+RF Fusion**:
- Dimensionality Reduction: Removes noise & redundant features
- Automatic Feature Learning: Learns features from data
- RF Speed: Fast & requires minimal hyperparameter tuning
- Combined Strengths: Leverages advantages of both methods

**Challenges**:
- Cross-dataset Performance: Difficult to generalize between years
- Class Imbalance: Normal data much more abundant than attack data
- Hyperparameter Tuning: Need to find optimal latent dimension

---

## Contributing & Support

For questions or improvements:
1. Check `notebooks/archived/` for previous experiments
2. Review reports in `results/experiments/`
3. Edit `src/config.py` to try different hyperparameters

---

## References

- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **CSE-CIC-IDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **mRMR Feature Selection**: https://github.com/ELELAB/pymrmr
- **PyTorch Autoencoder**: https://pytorch.org/tutorials/beginner/introyt/autoencoders_tutorial.html
- **scikit-learn Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests

---

**Created**: January 2026  
**Version**: 1.0

---

---

# Phiên Bản Tiếng Việt

## Mô Tả Dự Án

Dự án này phát triển một **hệ thống phát hiện xâm nhập mạng (IDS)** sử dụng kỹ thuật **kết hợp Autoencoder (AE) và Random Forest (RF)** để phát hiện và phân loại các cuộc tấn công mạng.

### Mục tiêu chính:
- **Giảm chiều dữ liệu**: Sử dụng Autoencoder để trích xuất đặc trưng potent từ dữ liệu lưu lượng mạng
- **Phân loại**: Sử dụng Random Forest để phân loại dữ liệu thành "Bình thường" (Benign) hoặc "Tấn công" (Attack)
- **So sánh phương pháp**: Đánh giá hiệu suất của AE+RF so với các phương pháp cơ sở (RF-only, SVM)
- **Kiểm tra cross-dataset**: Đánh giá khả năng tổng quát hóa trên các bộ dữ liệu khác nhau

---

## Cấu Trúc Dự Án

```
ids_ae_rf_fusion/
├── README.md                           # Tài liệu này
├── requirements.txt                    # Danh sách thư viện Python
├── setup_new_env.py                    # Script khởi tạo môi trường
│
├── src/                                # Mã nguồn chính
│   ├── config.py                       # Cấu hình dự án (đường dẫn, siêu tham số)
│   ├── autoencoder.py                  # Model Deep Autoencoder (PyTorch)
│   ├── rf_classifier.py                # Random Forest Classifier
│   ├── preprocessing.py                # Tiền xử lý và làm sạch dữ liệu
│   ├── feature_selection.py            # Lựa chọn đặc trưng (mRMR)
│   ├── evaluation.py                   # Đánh giá model & visualize kết quả
│   └── utils.py                        # Các hàm tiện ích
│
├── datasets/                           # Dữ liệu (chỉ chứa metadata)
│   └── dataset.txt                     # Mô tả bộ dữ liệu
│
├── notebooks/                          # Jupyter Notebooks (pipeline thực nghiệm)
│   ├── 0a_mRMR_selection.ipynb         # Lựa chọn đặc trưng (mRMR)
│   ├── 0b_mRMR_features_and_latent_features.ipynb
│   │
│   ├── 1a_ae_rf_fusion_mix_training.ipynb      # [Giai đoạn 1] Huấn luyện hỗn hợp
│   ├── 1b_rf_mix_training.ipynb                # Baseline: RF-only
│   ├── 1c_svm_mix_training.ipynb               # Baseline: SVM
│   │
│   ├── 2a_ae_rf_fusion_within_dataset.ipynb    # [Giai đoạn 2] Kiểm tra trong-dataset
│   ├── 2b_rf_within_dataset.ipynb
│   ├── 2c_svm_within_dataset.ipynb
│   │
│   ├── 3a_ae_rf_fusion_cross_dataset.ipynb     # [Giai đoạn 3] Kiểm tra cross-dataset
│   ├── 3b_rf_cross_dataset.ipynb
│   ├── 3c_svm_cross_dataset.ipynb
│   │
│   └── archived/                       # Các notebook cũ & thử nghiệm
│
├── results/                            # Kết quả thực nghiệm
│   ├── experiments/                    # Thư mục lưu kết quả từng lần chạy
│   │   └── exp_YYYYMMDD_HHMMSS/
│   │       ├── report_*.txt            # Báo cáo chi tiết (Accuracy, MCC, F1, ...)
│   │       ├── figures/                # Confusion Matrix & visualization
│   │       ├── models/                 # Mô hình đã huấn luyện
│   │       └── experiment_details.txt  # Cấu hình & siêu tham số
│   │
│   └── Summary/                        # Tóm tắt so sánh các phương pháp
│
└── models/                             # (Tùy chọn) Lưu các mô hình huấn luyện
```

---

## Yêu Cầu & Cài Đặt

### 1. **Yêu Cầu Hệ Thống**
- Python 3.8+
- CUDA (tùy chọn, để tăng tốc độ GPU nếu có)
- RAM: 8GB+ (khuyến nghị 16GB+)

### 2. **Cài Đặt Thư Viện**

```bash
# Cách 1: Cài đặt trực tiếp từ requirements.txt
pip install -r requirements.txt

# Cách 2: Sử dụng script setup (nếu có)
python setup_new_env.py
```

### 3. **Danh Sách Thư Viện**

Các thư viện chính được sử dụng:
- **numpy**, **pandas** - Xử lý dữ liệu
- **scikit-learn** - Machine Learning (Random Forest, SVM, metrics)
- **torch** - Deep Learning (Autoencoder)
- **matplotlib**, **seaborn** - Visualize kết quả
- **joblib** - Lưu & tải mô hình
- **mrmr-selection** - Lựa chọn đặc trưng

---

## Bộ Dữ Liệu

Dự án sử dụng hai bộ dữ liệu IDS công cộng:

| Bộ Dữ Liệu | Năm | Số mẫu | Đặc điểm |
|-----------|-----|--------|---------|
| **CIC-IDS2017** | 2017 | ~2.8M | Lưu lượng thực từ mạng, 15 loại tấn công |
| **CSE-CIC-IDS2018** | 2018 | ~2.5M | Cập nhật từ 2017, các tấn công hiện đại hơn |

### Các Loại Tấn Công:
- **BENIGN** - Lưu lượng bình thường
- **DOS** - Denial of Service (HULK, GoldenEye, SlowLoris, SlowHTTPTest)
- **DDOS** - Distributed DoS (HOIC, LOIC)
- **BRUTEFORCE** - Tấn công brute force (FTP, SSH)
- **BOT** - Botnet
- **PORTSCAN** - Quét cổng
- **WEB** - Tấn công Web (SQL Injection, XSS, Brute Force)
- **INFILTRATION** - Thâm nhập
- **HEARTBLEED** - Lỗ hổng Heartbleed

---

## Mô Tả Các Module Chính

### 1. **config.py** - Cấu Hình Toàn Dự Án

Quản lý tất cả các cấu hình:
- **Đường dẫn dữ liệu**: Thư mục datasets, results
- **Siêu tham số dữ liệu**:
  - `BINARY_MODE = True` - Chế độ phân loại nhị phân (Benign vs Attack)
  - `CHUNK_SIZE = 100000` - Kích thước đọc file mỗi lần
  - `SEED = 42` - Seed cho reproducibility

- **Tiền xử lý**:
  - `DROP_COLS` - Các cột loại bỏ (identifier, sparse, mismatch)
  - `RENAME_2018_TO_2017` - Map tên cột 2018 → 2017 (chuẩn hóa)
  - `BENIGN_LABELS` - Nhãn được coi là "bình thường"

- **Siêu tham số Autoencoder**:
  - Input dimension, latent dimension, hidden layers

- **Siêu tham số Random Forest**:
  - Số lượng cây (n_estimators)
  - Độ sâu tối đa (max_depth)
  - Cân bằng class weight

### 2. **autoencoder.py** - Deep Autoencoder

```python
class DeepAutoencoder(nn.Module):
    """
    Autoencoder đối xứng với:
    - Encoder: Input → Hidden Layers → Latent (bottleneck)
    - Decoder: Latent → Hidden Layers → Output
    
    Đặc điểm:
    - Batch Normalization + LeakyReLU
    - Output tuyến tính (linear activation)
    - Dùng MSE Loss để tái tạo dữ liệu
    """
```

**Chức năng**:
- Tái tạo (Reconstruction): Học trích xuất đặc trưng từ dữ liệu bình thường
- Giảm chiều: Nén 67 features xuống latent space nhỏ hơn
- Phát hiện: Các tấn công có lỗi tái tạo cao (anomaly detection)

### 3. **rf_classifier.py** - Random Forest Classifier

```python
def train_rf(X_train, y_train, save_path=None):
    """
    Huấn luyện Random Forest với:
    - n_estimators = 200
    - max_depth = 20
    - class_weight = 'balanced' (xử lý mất cân bằng dữ liệu)
    - n_jobs = 8 (đa xử lý)
    """
```

### 4. **preprocessing.py** - Tiền Xử Lý

- Đọc dữ liệu theo chunks
- Làm sạch dữ liệu (loại bỏ NaN, duplicates)
- Chuẩn hóa cột tên (2018 → 2017)
- Loại bỏ cột identifier (Flow ID, IP, Timestamp)
- Chuẩn hóa dữ liệu (StandardScaler)
- Mã hóa nhãn (Benign → 0, Attack → 1)

### 5. **feature_selection.py** - Lựa Chọn Đặc Trưng

Sử dụng **mRMR (Minimum Redundancy Maximum Relevance)**:
- Chọn đặc trưng có tương quan cao với nhãn
- Ưu tiên đặc trưng ít dự phòng

### 6. **evaluation.py** - Đánh Giá Mô Hình

Các metrics được tính toán:
- **Accuracy** - Độ chính xác tổng thể
- **MCC (Matthews Correlation Coefficient)** - Metric cân bằng cho dữ liệu mất cân bằng
- **Precision, Recall, F1-score** - Chi tiết từng class
- **Confusion Matrix** - Visualize true/false positives

---

## Hướng Dẫn Sử Dụng

### **Giai đoạn 0: Chuẩn Bị Dữ Liệu**
1. Tải CIC-IDS2017 & CSE-CIC-IDS2018 vào `datasets/CIC-IDS2017` và `datasets/CSE-CIC-IDS2018`
2. Chạy notebook `0a_mRMR_selection.ipynb` để lựa chọn đặc trưng
3. Kết quả: Danh sách ~20-30 đặc trưng tốt nhất

### **Giai đoạn 1: Huấn Luyện Hỗn Hợp (Mixed Training)**
- Gộp dữ liệu từ cả 2017 và 2018
- Huấn luyện: 80% / Kiểm tra: 20%
- Chạy:
  - `1a_ae_rf_fusion_mix_training.ipynb` - **Phương pháp đề xuất**
  - `1b_rf_mix_training.ipynb` - Baseline RF-only
  - `1c_svm_mix_training.ipynb` - Baseline SVM
- Kết quả: So sánh 3 phương pháp

### **Giai đoạn 2: Kiểm Tra Trong-Dataset (Within-Dataset)**
- Huấn luyện & kiểm tra trên cùng bộ dữ liệu:
  - 2017 train/test
  - 2018 train/test
- Chạy:
  - `2a_ae_rf_fusion_within_dataset.ipynb`
  - `2b_rf_within_dataset.ipynb`
  - `2c_svm_within_dataset.ipynb`

### **Giai đoạn 3: Kiểm Tra Cross-Dataset (Cross-Dataset)**
- Huấn luyện trên 1 bộ, kiểm tra trên bộ khác:
  - Train 2017 → Test 2018
  - Train 2018 → Test 2017
- Chạy:
  - `3a_ae_rf_fusion_cross_dataset.ipynb`
  - `3b_rf_cross_dataset.ipynb`
  - `3c_svm_cross_dataset.ipynb`
- **Mục đích**: Đánh giá khả năng tổng quát hóa

---

## Kết Quả Dự Kiến

### Hiệu Suất Dự Kiến
| Phương Pháp | Mixed | Within-2017 | Within-2018 | Cross (17→18) | Cross (18→17) |
|------------|-------|-------------|-------------|---------------|---------------|
| **AE+RF (Đề xuất)** | ~95-98% | ~97-99% | ~95-97% | ~85-90% | ~82-87% |
| **RF-only (Baseline)** | ~92-95% | ~95-98% | ~93-96% | ~80-85% | ~78-83% |
| **SVM (Baseline)** | ~90-94% | ~93-97% | ~91-95% | ~75-82% | ~73-80% |

**Ghi chú**: Cross-dataset thường có kết quả thấp hơn do sự khác biệt distribution giữa các bộ dữ liệu

---

## Cách Chạy Một Notebook

### 1. **Từ Command Line**
```bash
# Chạy notebook và lưu kết quả
jupyter nbconvert --to notebook --execute --output output.ipynb 1a_ae_rf_fusion_mix_training.ipynb

# Hoặc mở Jupyter Lab để chạy tương tác
jupyter lab
```

### 2. **Trong VS Code**
- Mở file `.ipynb`
- Chọn kernel Python
- Chạy từng cell hoặc Run All

### 3. **Thay đổi Cấu Hình**
- Chỉnh sửa `src/config.py`:
  - Đường dẫn dữ liệu
  - Siêu tham số mô hình
  - Feature selection parameters
- Cấu hình tự động được load trong notebooks

---

## Cấu Trúc Kết Quả (Results)

Mỗi lần chạy notebook tạo thư mục kết quả:
```
results/experiments/exp_YYYYMMDD_HHMMSS/
├── experiment_details.txt          # Cấu hình, siêu tham số
├── report_Mixed_Test_Set.txt       # Báo cáo metrics chi tiết
├── figures/
│   ├── cm_Mixed_Test_Set.png       # Confusion Matrix
│   └── ... (các hình khác)
└── models/
    └── model.joblib / model.pt     # Mô hình đã huấn luyện
```

---

## Những Điểm Chính

**Ưu điểm của AE+RF Fusion**:
- Giảm chiều: Loại bỏ features noise & redundant
- Học được đặc trưng tự động từ dữ liệu
- RF nhanh & không cần điều chỉnh siêu tham số phức tạp
- Kết hợp: Lợi thế của cả 2 phương pháp

**Thách thức**:
- Cross-dataset performance: Mô hình khó tổng quát hóa giữa các năm
- Class imbalance: Dữ liệu bình thường nhiều hơn dữ liệu tấn công
- Hyperparameter tuning: Cần tìm optimal latent dimension

---

## Đóng Góp & Hỗ Trợ

Nếu có câu hỏi hoặc muốn cải tiến dự án:
1. Kiểm tra `notebooks/archived/` để xem các thử nghiệm trước đó
2. Tham khảo các báo cáo trong `results/experiments/`
3. Chỉnh sửa `src/config.py` để thử các siêu tham số khác nhau

---

## Tham Khảo

- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **CSE-CIC-IDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **mRMR Feature Selection**: https://github.com/ELELAB/pymrmr
- **PyTorch Autoencoder**: https://pytorch.org/tutorials/beginner/introyt/autoencoders_tutorial.html
- **scikit-learn Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests

---

**Ngày tạo**: January 2026  
**Phiên bản**: 1.0
