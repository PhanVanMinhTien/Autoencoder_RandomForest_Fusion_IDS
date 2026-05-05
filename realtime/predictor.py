# predictor.py

import argparse
import json
import os
import time
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


FEATURE_30 = [
    "RST Flag Count",
    "Total Length of Fwd Packets",
    "Bwd IAT Min",
    "ECE Flag Count",
    "act_data_pkt_fwd",
    "Idle Std",
    "Bwd Packet Length Min",
    "Total Fwd Packets",
    "Bwd IAT Mean",
    "PSH Flag Count",
    "Destination Port",
    "Flow IAT Std",
    "Bwd Packet Length Std",
    "Bwd IAT Max",
    "Fwd Packet Length Max",
    "Fwd PSH Flags",
    "Active Min",
    "Init_Win_bytes_backward",
    "SYN Flag Count",
    "Flow Duration",
    "Fwd IAT Min",
    "Down/Up Ratio",
    "Bwd IAT Std",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Bwd Packets/s",
    "Active Mean",
    "Fwd IAT Mean",
    "URG Flag Count",
    "Min Packet Length",
]


class DeepAutoencoder(nn.Module):
    """
    Full Autoencoder architecture.
    The encoder is used to extract 5 latent features.
    Architecture must match Phase 1 training.
    """

    def __init__(self, input_dim=30, latent_dim=5, hidden_layers=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [22, 12]

        encoder_layers = []
        last_dim = input_dim

        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            last_dim = h_dim

        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        last_dim = latent_dim

        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            last_dim = h_dim

        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class AERFPredictor:
    def __init__(
        self,
        scaler_path,
        ae_path,
        rf_path,
        block_url,
        idle_timeout=300,
        dry_run=False
    ):
        self.scaler_path = scaler_path
        self.ae_path = ae_path
        self.rf_path = rf_path
        self.block_url = block_url
        self.idle_timeout = idle_timeout
        self.dry_run = dry_run

        self.blocked_ips = set()

        print("[*] Loading scaler...")
        self.scaler = joblib.load(self.scaler_path)

        print("[*] Loading Random Forest...")
        self.rf_model = joblib.load(self.rf_path)

        print("[*] Loading Autoencoder...")
        self.ae = DeepAutoencoder(input_dim=30, latent_dim=5, hidden_layers=[22, 12])
        state = torch.load(self.ae_path, map_location=torch.device("cpu"))

        # Support common checkpoint formats
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing AE keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected AE keys: {unexpected}")

        self.ae.eval()

        print("[OK] Predictor loaded successfully.")

    def build_input_df(self, row):
        values = []

        for feature in FEATURE_30:
            value = row.get(feature, 0)

            if pd.isna(value):
                value = 0

            values.append(value)

        return pd.DataFrame([values], columns=FEATURE_30)

    def predict_row(self, row):
        df_input = self.build_input_df(row)

        # 1. Scale 30 selected features
        X_scaled = self.scaler.transform(df_input)

        # 2. AE encoder -> 5 latent features
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            X_latent = self.ae.encoder(X_tensor).numpy()

        # 3. Fusion: first 20 scaled features + 5 latent features = 25
        X_fusion = np.hstack([X_scaled[:, :20], X_latent])

        # 4. RF prediction
        pred = self.rf_model.predict(X_fusion)[0]

        return pred

    def is_attack(self, pred):
        """
        Handle different label formats.
        Adjust this if your model uses other labels.
        """
        if isinstance(pred, str):
            return pred.upper() in ["ATTACK", "MALICIOUS", "SUSPECT", "ANOMALY"]

        return int(pred) == 1

    def send_block_request(self, src_ip):
        if not src_ip or src_ip in self.blocked_ips:
            return

        self.blocked_ips.add(src_ip)

        payload = {
            "src_ip": src_ip,
            "idle_timeout": self.idle_timeout
        }

        print(f"[IPS] Sending block request: {payload}")

        if self.dry_run:
            print("[DRY-RUN] Block request not sent.")
            return

        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.block_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                body = response.read().decode("utf-8")
                print(f"[IPS] Controller response: {body}")

        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to send block request: {e}")


def monitor_csv(csv_file, predictor, interval=1.0):
    seen = 0

    print(f"[*] Monitoring CSV: {csv_file}")
    print("[*] Press Ctrl+C to stop.")

    while True:
        if not os.path.exists(csv_file):
            time.sleep(interval)
            continue

        try:
            df = pd.read_csv(csv_file)

            # If file was recreated/truncated
            if len(df) < seen:
                seen = 0

            if len(df) > seen:
                new_rows = df.iloc[seen:]

                for _, row in new_rows.iterrows():
                    flow_id = row.get("Flow ID", "unknown-flow")
                    src_ip = row.get("Source IP", None)
                    dst_ip = row.get("Destination IP", None)

                    try:
                        pred = predictor.predict_row(row)
                        attack = predictor.is_attack(pred)

                        if attack:
                            print(f"[ATTACK] {flow_id} | {src_ip} -> {dst_ip} | pred={pred}")
                            predictor.send_block_request(src_ip)
                        else:
                            print(f"[NORMAL] {flow_id} | {src_ip} -> {dst_ip} | pred={pred}")

                    except Exception as e:
                        print(f"[ERROR] Prediction failed for {flow_id}: {e}")

                seen = len(df)

        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="collected_data.csv")
    parser.add_argument("--scaler", default="model/scaler.joblib")
    parser.add_argument("--ae", default="model/autoencoder.pth")
    parser.add_argument("--rf", default="model/rf_model.pkl")
    parser.add_argument("--block-url", default="http://127.0.0.1:8080/block")
    parser.add_argument("--idle-timeout", type=int, default=300)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    predictor = AERFPredictor(
        scaler_path=args.scaler,
        ae_path=args.ae,
        rf_path=args.rf,
        block_url=args.block_url,
        idle_timeout=args.idle_timeout,
        dry_run=args.dry_run
    )

    try:
        monitor_csv(args.csv, predictor, interval=args.interval)
    except KeyboardInterrupt:
        print("\n[!] Predictor stopped.")