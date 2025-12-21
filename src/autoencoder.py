# src/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from . import config

import multiprocessing


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=config.AE_INPUT_DIM, latent_dim=config.AE_LATENT_DIM, hidden_layers=config.AE_HIDDEN_LAYERS):
        super(DeepAutoencoder, self).__init__()
        
        # --- 1. BUILD ENCODER (Tự động) ---
        encoder_modules = []
        
        # Layer đầu tiên: Input -> Hidden 1 (Ví dụ: 20 -> 16)
        current_dim = input_dim
        for h_dim in hidden_layers:
            encoder_modules.append(nn.Linear(current_dim, h_dim))
            encoder_modules.append(nn.BatchNorm1d(h_dim)) 
            encoder_modules.append(nn.LeakyReLU(0.2)) 
            current_dim = h_dim
        
        # Layer cuối cùng của Encoder: Hidden Cuối -> Latent (Ví dụ: 12 -> 10)
        encoder_modules.append(nn.Linear(current_dim, latent_dim))
        #encoder_modules.append(nn.ReLU()) # Latent Z (Bottleneck)
        self.encoder = nn.Sequential(*encoder_modules)

        
        # --- 2. BUILD DECODER (Đối xứng ngược lại) ---
        decoder_modules = []
        
        # Layer đầu tiên của Decoder: Latent -> Hidden Cuối (Ví dụ: 10 -> 12)
        current_dim = latent_dim
        # Duyệt ngược danh sách hidden (Ví dụ: [16, 12] -> đảo thành [12, 16])
        reversed_hidden = hidden_layers[::-1]
        
        for h_dim in reversed_hidden:
            decoder_modules.append(nn.Linear(current_dim, h_dim))
            decoder_modules.append(nn.BatchNorm1d(h_dim))
            decoder_modules.append(nn.LeakyReLU(0.2))
            current_dim = h_dim
            
        # Layer cuối cùng: Hidden 1 -> Output (Ví dụ: 16 -> 20)
        decoder_modules.append(nn.Linear(current_dim, input_dim))
        # Lưu ý: Output không dùng activation (hoặc dùng Sigmoid nếu data 0-1)
        # Ở đây data đã StandardScaler nên output là tuyến tính (linear)
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def get_latent(self, x):
        with torch.no_grad():
            self.eval()
            return self.encoder(x)

def train_ae(model, X_train, save_path=None):
    """Hàm train AE"""
    print(f"[Autoencoder] Training on {config.DEVICE}...")
    model.to(config.DEVICE)
    
    # Sử dụng đa luồng CPU để load data nhanh hơn
    num_cpus = multiprocessing.cpu_count()
    workers = min(8, num_cpus)  # Giới hạn số worker để tránh quá tải hệ thống

    # Prepare Data
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)



    # loader = DataLoader(dataset, batch_size=config.AE_BATCH_SIZE, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=config.AE_BATCH_SIZE, 
        shuffle=True,
        num_workers=8,  # <--- Dùng đa luồng CPU để đọc data song song
        pin_memory=False       # <--- Giúp truyền data từ RAM lên GPU nhanh hơn
    )
    
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.AE_LR)
    
    # Training Loop
    model.train()
    for epoch in range(config.AE_EPOCHS):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(config.DEVICE)
            
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config.AE_EPOCHS} - Loss: {total_loss/len(loader):.6f}")    

    #Save Model
    if save_path:
        # Nếu có đường dẫn truyền vào (từ Experiment), lưu vào đó
        torch.save(model.state_dict(), save_path)
        print(f"✅ AE Model saved to: {save_path}")
    else:
        # Fallback: Nếu không truyền, lưu vào chỗ cũ (mặc định)
        default_path = config.MODELS_DIR / "autoencoder/full_ae_model.pth"
        torch.save(model.state_dict(), default_path)
        print(f"⚠️ AE Model saved to default: {default_path}")
    # 4. Lưu Model
    # if save_path:
    #     # Trường hợp có truyền save_path (Ví dụ từ Grid Search)
    #     save_path = Path(save_path)
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(model.state_dict(), save_path)
    #     print(f"✅ AE Model saved to: {save_path}")
    # else:
    #     # Trường hợp save_path=None -> Lưu vào fallback default
    #     default_path = config.MODELS_DIR / "autoencoder/full_ae_model.pth"
        
    #     # --- QUAN TRỌNG: Thêm dòng này để tạo folder models/autoencoder ---
    #     default_path.parent.mkdir(parents=True, exist_ok=True)
    #     # -----------------------------------------------------------------
        
    #     torch.save(model.state_dict(), default_path)
    #     print(f"⚠️ AE Model saved to default: {default_path}")
        
    return model


def train_ae_v2(model, X_train, val_split=0.1, epochs=50, batch_size=1024, lr=0.001):
    # 1. Chia Validation set để theo dõi hiện tượng Overfitting/Divergence
    train_idx, val_idx = train_test_split(range(len(X_train)), test_size=val_split, stratify=None)
    
    # 2. Setup Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Tự động giảm LR nếu loss không giảm sau 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    # Biến cho Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7

    # [Tiến hành training loop...]
    # Trong loop, sau mỗi epoch:
    # val_loss = evaluate_on_val_set(model, X_val)
    # scheduler.step(val_loss)
    
    # Nếu val_loss tăng liên tục -> Dừng sớm để tránh nổ Gradient






def extract_features(model, X):
    """Hàm helper để chạy data qua Encoder lấy Z"""
    model.to(config.DEVICE)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(config.DEVICE)
    Z = model.get_latent(X_tensor)
    return Z.cpu().numpy()