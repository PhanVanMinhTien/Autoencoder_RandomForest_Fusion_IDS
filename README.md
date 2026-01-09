# IDS AE+RF Fusion - Network Intrusion Detection System

## 📋 Mô Tả Dự Án

Dự án này phát triển một **hệ thống phát hiện xâm nhập mạng (IDS)** sử dụng kỹ thuật **kết hợp Autoencoder (AE) và Random Forest (RF)** để phát hiện và phân loại các cuộc tấn công mạng.

### Mục tiêu chính:
- **Giảm chiều dữ liệu**: Sử dụng Autoencoder để trích xuất đặc trưng potent từ dữ liệu lưu lượng mạng
- **Phân loại**: Sử dụng Random Forest để phân loại dữ liệu thành "Bình thường" (Benign) hoặc "Tấn công" (Attack)
- **So sánh phương pháp**: Đánh giá hiệu suất của AE+RF so với các phương pháp cơ sở (RF-only, SVM)
- **Kiểm tra cross-dataset**: Đánh giá khả năng tổng quát hóa trên các bộ dữ liệu khác nhau

---

## 📁 Cấu Trúc Dự Án

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

## 🛠️ Yêu Cầu & Cài Đặt

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

## 📊 Bộ Dữ Liệu

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

## 🔧 Mô Tả Các Module Chính

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

## 🚀 Hướng Dẫn Sử Dụng

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

## 📈 Kết Quả Dự Kiến

### Hiệu Suất Dự Kiến
| Phương Pháp | Mixed | Within-2017 | Within-2018 | Cross (17→18) | Cross (18→17) |
|------------|-------|-------------|-------------|---------------|---------------|
| **AE+RF (Đề xuất)** | ~95-98% | ~97-99% | ~95-97% | ~85-90% | ~82-87% |
| **RF-only (Baseline)** | ~92-95% | ~95-98% | ~93-96% | ~80-85% | ~78-83% |
| **SVM (Baseline)** | ~90-94% | ~93-97% | ~91-95% | ~75-82% | ~73-80% |

**Ghi chú**: Cross-dataset thường có kết quả thấp hơn do sự khác biệt distribution giữa các bộ dữ liệu

---

## 📝 Cách Chạy Một Notebook

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

## 🔍 Cấu Trúc Kết Quả (Results)

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

## 🎯 Những Điểm Chính

✅ **Ưu điểm của AE+RF Fusion**:
- Giảm chiều: Loại bỏ features noise & redundant
- Học được đặc trưng tự động từ dữ liệu
- RF nhanh & không cần điều chỉnh siêu tham số phức tạp
- Kết hợp: Lợi thế của cả 2 phương pháp

⚠️ **Thách thức**:
- Cross-dataset performance: Mô hình khó tổng quát hóa giữa các năm
- Class imbalance: Dữ liệu bình thường nhiều hơn dữ liệu tấn công
- Hyperparameter tuning: Cần tìm optimal latent dimension

---

## 🤝 Đóng Góp & Hỗ Trợ

Nếu có câu hỏi hoặc muốn cải tiến dự án:
1. Kiểm tra `notebooks/archived/` để xem các thử nghiệm trước đó
2. Tham khảo các báo cáo trong `results/experiments/`
3. Chỉnh sửa `src/config.py` để thử các siêu tham số khác nhau

---

## 📚 Tham Khảo

- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **CSE-CIC-IDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **mRMR Feature Selection**: https://github.com/ELELAB/pymrmr
- **PyTorch Autoencoder**: https://pytorch.org/tutorials/beginner/introyt/autoencoders_tutorial.html
- **scikit-learn Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests

---

**Ngày tạo**: January 2026  
**Phiên bản**: 1.0
