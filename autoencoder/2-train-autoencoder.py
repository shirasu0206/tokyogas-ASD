import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. Dataset の定義
# ---------------------------
class AudioDataset(Dataset):
    def __init__(self, file_list, sr=16000, n_mels=128, desired_width=311):
        self.file_list = file_list
        self.sr = sr
        self.n_mels = n_mels
        self.desired_width = desired_width

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        y, sr = librosa.load(file_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=1024, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = np.clip((mel_spec_db + 80) / 80, 0, 1)

        current_width = mel_spec_norm.shape[1]
        if current_width < self.desired_width:
            pad_width = self.desired_width - current_width
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        elif current_width > self.desired_width:
            mel_spec_norm = mel_spec_norm[:, :self.desired_width]

        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=0)
        return torch.tensor(mel_spec_norm, dtype=torch.float32)

# ---------------------------
# 2. オートエンコーダの定義
# ---------------------------
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---------------------------
# 3. 学習ループ
# ---------------------------
def main():
    train_dir = "./data/pitch-train"
    valid_dir = "./data/pitch-valid"
    
    train_files = glob.glob(os.path.join(train_dir, "*.wav"))
    valid_files = glob.glob(os.path.join(valid_dir, "*.wav"))
    print(f"訓練データ: {len(train_files)} ファイル, 検証データ: {len(valid_files)} ファイル")
    
    desired_width = 311
    train_dataset = AudioDataset(train_files, sr=16000, n_mels=128, desired_width=desired_width)
    valid_dataset = AudioDataset(valid_files, sr=16000, n_mels=128, desired_width=desired_width)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    model = AudioAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs} 開始")

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

            if batch_idx % 10 == 0:
                print(f"  🟢 バッチ {batch_idx+1}/{len(train_loader)} | 損失: {loss.item():.6f}")

        train_loss = running_loss / len(train_dataset)
        print(f"✅ Epoch {epoch+1} | 訓練損失: {train_loss:.6f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                valid_loss += loss.item() * batch.size(0)

                if batch_idx % 5 == 0:
                    print(f"  🟠 検証バッチ {batch_idx+1}/{len(valid_loader)} | 検証損失: {loss.item():.6f}")

        valid_loss /= len(valid_dataset)
        print(f"🟡 Epoch {epoch+1} | 検証損失: {valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_autoencoder.pth")
            print("🎉 → ベストモデルを保存しました！")

    print("✅ 学習完了！")

if __name__ == '__main__':
    main()
