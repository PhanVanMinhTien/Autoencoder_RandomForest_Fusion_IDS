import pandas as pd
import joblib
import time
import os

MODEL_PATH = "model/rf_model.pkl"
CSV_FILE = "collected_data.csv"

model = joblib.load(MODEL_PATH)
seen = 0

FEATURE_30 = [
    'RST Flag Count', 
    'Total Length of Fwd Packets', 
    'Bwd IAT Min', 
    'ECE Flag Count',
    'act_data_pkt_fwd',
    'Idle Std', 
    'Bwd Packet Length Min', 
    'Total Fwd Packets', 
    'Bwd IAT Mean', 
    'PSH Flag Count', 
    'Destination Port', 
    'Flow IAT Std', 
    'Bwd Packet Length Std', 
    'Bwd IAT Max', 
    'Fwd Packet Length Max', 
    'Fwd PSH Flags', 
    'Active Min', 
    'Init_Win_bytes_backward', 
    'SYN Flag Count', 
    'Flow Duration', 
    'Fwd IAT Min', 
    'Down/Up Ratio', 
    'Bwd IAT Std', 
    'Fwd Packet Length Std',
    'Fwd IAT Total', 
    'Bwd Packets/s', 
    'Active Mean', 
    'Fwd IAT Mean', 
    'URG Flag Count', 
    'Min Packet Length'
]

def build_feature_vector(row):
    return [row.get(f, 0) for f in FEATURE_30]

print("Detecting...")

while True:
    if not os.path.exists(CSV_FILE):
        time.sleep(1)
        continue

    df = pd.read_csv(CSV_FILE)

    if len(df) > seen:
        new_rows = df.iloc[seen:]

        for _, row in new_rows.iterrows():
            features = build_feature_vector(row)

            if len(features) != 30:
                continue

            pred = model.predict([features])[0]

            if pred == 1:
                print(f"ATTACK: {row['Flow ID']}")
            else:
                print(f"NORMAL: {row['Flow ID']}")

        seen = len(df)

    time.sleep(1)