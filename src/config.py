# src/config.py
import torch
from pathlib import Path

# --- PATHS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_2017 = ROOT_DIR / "datasets" / "CIC-IDS2017"
DIR_2018 = ROOT_DIR / "datasets" / "CSE-CIC-IDS2018"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

# Tạo folder nếu chưa có
# FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# (MODELS_DIR / "autoencoder").mkdir(parents=True, exist_ok=True)
# (MODELS_DIR / "classifier").mkdir(parents=True, exist_ok=True)

# --- DATA CONFIG ---
BINARY_MODE = True
CHUNK_SIZE = 100000 # Kích thước đọc file mỗi lần
SEED = 42           # 16 37 42 

# --- CLEANING CONFIG (FINAL) ---

BENIGN_LABELS = ['benign', 'normal']

# Danh sách các cột cần loại bỏ: Identifier + Duplicate + Sparse + Mismatch
DROP_COLS = [
    # 1. Identifiers (Định danh - Gây overfitting)
    "Flow ID", 
    "Src IP", "Dst IP", "Source IP", "Destination IP", 
    "Timestamp", "SimillarHTTP", 
    
    # 2. Error Column (Lỗi chỉ có ở 2017)
    "Fwd Header Length.1", 
    
    # 3. Feature bị lệch (Mismatch giữa 2 bộ)
    "Protocol", 
    
    # 4. Redundant/Sparse (Tên kiểu 2017 - Full Name)
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    
    # 5. Redundant/Sparse (Tên kiểu 2018 - Abbreviated Name)
    "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"
]

# --- COLUMN MAPPING (2018 -> 2017) ---
RENAME_2018_TO_2017 = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward"
}

# --- LABEL MAPPING ---
# RAW_TO_CANONICAL = {
#     "BENIGN": "BENIGN",
#     "DOS HULK": "DOS", 
#     "DOS GOLDENEYE": "DOS", 
#     "DOS SLOWLORIS": "DOS", 
#     "DOS SLOWHTTPTEST": "DOS",
#     "DOS ATTACK-HULK": "DOS", 
#     "DOS ATTACK-GOLDENEYE": "DOS", 
#     "DOS ATTACK-SLOWLORIS": "DOS", 
#     "DOS ATTACK-SLOWHTTPTEST": "DOS",
#     "DDOS": "DDOS", 
#     "DDOS ATTACK-HOIC": "DDOS", 
#     "DDOS ATTACK-LOIC-UDP": "DDOS", 
#     "DDOS ATTACK-LOIC-HTTP": "DDOS",
#     "FTP-PATATOR": "BRUTEFORCE", 
#     "SSH-PATATOR": "BRUTEFORCE",
#     "BOT": "BOT", 
#     "PORTSCAN": "PORTSCAN", 
#     "INFILTRATION": "INFILTRATION",
#     "WEB ATTACK - BRUTE FORCE": "WEB", 
#     "WEB ATTACK - XSS": "WEB", 
#     "WEB ATTACK - SQL INJECTION": "WEB"
# }

RAW_TO_CANONICAL = {
    # 1. BENIGN
    "BENIGN": "BENIGN",

    # 2. DOS (Gom cả 2017 và 2018)
    "DOS HULK": "DOS",
    "DOS GOLDENEYE": "DOS",
    "DOS SLOWLORIS": "DOS",
    "DOS SLOWHTTPTEST": "DOS",
    # Lưu ý 2018 dùng "ATTACKS" (số nhiều) thay vì "ATTACK"
    "DOS ATTACKS-HULK": "DOS",
    "DOS ATTACKS-GOLDENEYE": "DOS",
    "DOS ATTACKS-SLOWLORIS": "DOS",
    "DOS ATTACKS-SLOWHTTPTEST": "DOS",

    # 3. DDOS
    "DDOS": "DDOS",
    "DDOS ATTACK-HOIC": "DDOS",
    "DDOS ATTACK-LOIC-UDP": "DDOS",     # Một số bản ghi 2018
    "DDOS ATTACKS-LOIC-HTTP": "DDOS",   # 2018 dùng số nhiều

    # 4. BOTNET
    "BOT": "BOT",

    # 5. PORTSCAN (Chủ yếu 2017)
    "PORTSCAN": "PORTSCAN",

    # 6. BRUTE FORCE (FTP/SSH) - Gom nhóm mạng
    "FTP-PATATOR": "BRUTEFORCE",        # 2017
    "SSH-PATATOR": "BRUTEFORCE",        # 2017
    "FTP-BRUTEFORCE": "BRUTEFORCE",     # 2018
    "SSH-BRUTEFORCE": "BRUTEFORCE",     # 2018

    # 7. INFILTRATION (Lưu ý lỗi chính tả 2018)
    "INFILTRATION": "INFILTRATION",     # 2017
    "INFILTERATION": "INFILTRATION",    # 2018 (Sai chính tả gốc trong dataset)

    # 8. WEB ATTACKS (Gom nhóm Web)
    # 2018
    "BRUTE FORCE -WEB": "WEB",
    "BRUTE FORCE -XSS": "WEB",
    "SQL INJECTION": "WEB",
    # 2017 (Có ký tự lạ hoặc khoảng trắng)
    "WEB ATTACK - BRUTE FORCE": "WEB",
    "WEB ATTACK - XSS": "WEB",
    "WEB ATTACK - SQL INJECTION": "WEB",
    # Map thêm trường hợp ký tự lạ  (copy từ log scan) để chắc chắn bắt được
    "WEB ATTACK  BRUTE FORCE": "WEB",
    "WEB ATTACK  XSS": "WEB",
    "WEB ATTACK  SQL INJECTION": "WEB",

    # 9. OTHER (Heartbleed)
    "HEARTBLEED": "HEARTBLEED"
}

# --- SELECTED FEATURES ---




# 67
#                      'Destination Port', 'Flow Duration', 'Total Fwd Packets', 
#                      'Total Backward Packets', 'Total Length of Fwd Packets', 
#                      'Total Length of Bwd Packets', 'Fwd Packet Length Max', 
#                      'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 
#                      'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 
#                      'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 
#                      'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 
#                      'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 
#                      'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 
#                      'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 
#                      'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 
#                      'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 
#                      'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 
#                      'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 
#                      'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 
#                      'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Init_Win_bytes_forward', 
#                      'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 
#                      'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 
#                      'Idle Std', 'Idle Max', 'Idle Min'


# 65
SELECTED_FEATURES = ['RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 'ECE Flag Count', 
                     'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 'Total Fwd Packets', 
                     'Bwd IAT Mean', 'PSH Flag Count', 'Destination Port', 'Flow IAT Std', 'Bwd Packet Length Std', 
                     'Bwd IAT Max', 'Active Min', 'Fwd PSH Flags', 'Fwd Packet Length Max', 'Init_Win_bytes_backward', 
                     'Flow Duration', 'SYN Flag Count', 'Fwd IAT Min', 'Bwd IAT Std', 'Down/Up Ratio', 
                     'Fwd Header Length', 'Fwd IAT Total', 'Active Mean', 'Fwd Packet Length Std', 
                     'Fwd IAT Mean', 'URG Flag Count', 'Min Packet Length', 'Idle Max', 'Bwd Packets/s', 
                     'Packet Length Std', 'Flow IAT Max', 'Fwd Packet Length Mean', 'ACK Flag Count', 'Fwd IAT Max', 
                     'Avg Fwd Segment Size', 'Idle Mean', 'Fwd Packet Length Min', 'Active Max', 'Idle Min', 
                     'Flow IAT Min', 'Fwd Packets/s', 'Bwd Packet Length Max', 'Init_Win_bytes_forward', 'Active Std',
                     'FIN Flag Count', 'Max Packet Length', 'Bwd IAT Total', 'Flow IAT Mean', 'Flow Packets/s', 
                     'Flow Bytes/s', 'Total Backward Packets', 'Fwd URG Flags', 'Average Packet Size', 
                     'Total Length of Bwd Packets', 'min_seg_size_forward', 'Packet Length Mean', 'Bwd Header Length', 
                     'CWE Flag Count', 'Packet Length Variance', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Fwd IAT Std']

#40
AE_INPUT_FEATIRES = ['RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 'ECE Flag Count', 
                     'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 'Total Fwd Packets', 'Bwd IAT Mean', 
                     'PSH Flag Count', 'Destination Port', 'Flow IAT Std', 'Bwd Packet Length Std', 'Bwd IAT Max', 
                     'Fwd PSH Flags', 'Fwd Packet Length Max', 'Active Min', 'Init_Win_bytes_backward', 'Bwd Packets/s', 
                     'SYN Flag Count', 'Fwd IAT Min', 'Flow Duration', 'Down/Up Ratio', 'Bwd IAT Std', 'Fwd IAT Mean', 
                     'Fwd Packet Length Std', 'Active Mean', 'Fwd IAT Total', 'Min Packet Length', 'URG Flag Count', 
                     'Idle Max', 'Fwd Packet Length Mean', 'Flow IAT Max', 'Packet Length Std', 'Avg Fwd Segment Size', 
                     'Fwd IAT Max', 'ACK Flag Count', 'Idle Mean', 'Fwd Packet Length Min', 'Idle Min']

#20
mRMR_FEATURES = [
                'RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 
                'ECE Flag Count', 'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 
                'Total Fwd Packets', 'Bwd IAT Mean', 'PSH Flag Count', 'Destination Port', 
                'Flow IAT Std', 'Bwd Packet Length Std', 'Bwd IAT Max', 'Fwd Packet Length Max', 
                'Fwd PSH Flags', 'Active Min', 'Init_Win_bytes_backward', 'Bwd Packets/s', 'Fwd IAT Min'
                ]

# 32
# mRMR_FEATURES = ['RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 
#                      'ECE Flag Count', 'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 
#                      'Total Fwd Packets', 'Bwd IAT Mean', 'PSH Flag Count', 'Destination Port', 
#                      'Flow IAT Std', 'Bwd Packet Length Std', 'Bwd IAT Max', 'Fwd Packet Length Max', 
#                      'Fwd PSH Flags', 'Active Min', 'Init_Win_bytes_backward', 'SYN Flag Count', 
#                      'Flow Duration', 'Fwd IAT Min', 'Down/Up Ratio', 'Bwd IAT Std', 'Fwd Packet Length Std', 
#                      'Fwd IAT Total', 'Bwd Packets/s', 'Active Mean', 'Fwd IAT Mean', 'URG Flag Count', 
#                      'Min Packet Length', 'Idle Max', 'Fwd Packet Length Mean']

# --- MODEL HYPERPARAMETERS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Deep Autoencoder 

# AE_INPUT_DIM = len(SELECTED_FEATURES)
# AE_LATENT_DIM = 10
# AE_HIDDEN_LAYERS = [64, 32]
# AE_EPOCHS = 20
# AE_BATCH_SIZE = 1024
# AE_LR = 1e-3


# AE_INPUT_DIM = len(SELECTED_FEATURES) # = 20
# AE_LATENT_DIM = 10 
# AE_HIDDEN_LAYERS = [16, 12] 
# AE_EPOCHS = 20

AE_INPUT_DIM = 30 
AE_LATENT_DIM = 5
AE_HIDDEN_LAYERS = [22, 12] 
AE_EPOCHS = 50


# Thay đổi từ 1024 lên hẳn 16384 hoặc 32768
AE_BATCH_SIZE = 16384
AE_LR = 0.0005

# Random Forest

RF_ESTIMATORS = 100         # 50 100 200
RF_MAX_DEPTH = 12           # 8 10 12 15



# SVM 
RANDOM_STATE = 42