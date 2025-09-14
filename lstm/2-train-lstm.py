import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------
# 1) Dataset 定義: LSTM の入力に合わせた形に変換
# -----------------------------------------------------
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

        # メルスペクトログラムを計算: shape = (n_mels, time_frames)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            n_fft=1024, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # [0,1] に正規化 (-80 dB ~ 0 dB を 0~1 にマップ)
        mel_spec_norm = np.clip((mel_spec_db + 80) / 80, 0, 1)

        # 幅 (time_frames) を desired_width に揃える
        # 短ければ右側を0パディング, 長ければカット
        current_width = mel_spec_norm.shape[1]
        if current_width < self.desired_width:
            pad_width = self.desired_width - current_width
            mel_spec_norm = np.pad(
                mel_spec_norm, 
                pad_width=((0,0), (0,pad_width)), 
                mode='constant'
            )
        else:
            mel_spec_norm = mel_spec_norm[:, :self.desired_width]

        # (n_mels, desired_width) -> (desired_width, n_mels) に転置
        # これは LSTM における [time_steps, feature_dim] に相当
        mel_spec_norm = mel_spec_norm.T  # shape: (311, 128)

        # Tensor化して返す
        return torch.tensor(mel_spec_norm, dtype=torch.float32)


# -----------------------------------------------------
# 2) LSTM Autoencoder の定義
#    - Encoder と Decoder を分けて書く例も多い
#    - ここでは一つのクラスでまとめています
# -----------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=32, num_layers=1):
        """
        input_dim:  入力の特徴次元 (例: メルスペクトログラムの周波数数)
        hidden_dim: LSTMの隠れ状態次元
        latent_dim: 潜在表現の次元 (Decoderへ渡すために hidden_dim -> latent_dim などの線形変換を想定)
        num_layers: LSTMスタック数
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  
            bidirectional=False
        )
        
        # hidden_dim -> latent_dim の線形変換 (任意)
        self.enc_linear = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        # Decoder の入力次元は latent_dim にしてもよいし、
        # hidden_dim に戻して LSTMへ渡す形でも可
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # hidden_dim -> input_dim に戻す出力層
        self.dec_output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        x: shape = (batch_size, time_steps, input_dim)
        """
        # ------- Encoder -------
        # LSTMエンコーダに入力し、最終ステップの hidden state を潜在表現とみなす
        # out: shape (batch, time_steps, hidden_dim)
        # (h_n, c_n): shape (num_layers, batch, hidden_dim)
        out, (h_n, c_n) = self.encoder_lstm(x)

        # h_n[-1] (最後のlayerの hidden state) を取り出し、latent_dimに圧縮
        # h_n: (num_layers, batch, hidden_dim)
        # ここでは最上層 (num_layers-1) を用いる
        h_enc = h_n[-1]  # shape: (batch, hidden_dim)
        
        z = self.enc_linear(h_enc)  # shape: (batch, latent_dim)
        
        # ------- Decoder -------
        # デコーダでは、time_steps 分だけ出力を生成する必要があるので、
        # 簡易的には "z を各タイムステップに繰り返し" 入力する、などの方法があります。
        # ここでは z を time_steps 回複製して LSTMに投入する例とします。
        
        batch_size, time_steps, _ = x.size()
        
        # (batch_size, time_steps, latent_dim)
        z_repeated = z.unsqueeze(1).repeat(1, time_steps, 1)
        
        # LSTM decoder
        dec_out, (dec_h, dec_c) = self.decoder_lstm(z_repeated)
        
        # 最後に dec_out を (batch_size, time_steps, input_dim) に線形変換
        # dec_out: shape (batch_size, time_steps, hidden_dim)
        x_recon = self.dec_output(dec_out)  # shape: (batch_size, time_steps, input_dim)

        return x_recon


# -----------------------------------------------------
# 3) 学習ループ & 検証ループ
# -----------------------------------------------------
def train_lstm_autoencoder(
    model, train_loader, valid_loader, 
    device, num_epochs=50, lr=1e-3
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)  # shape: (B, T, F)
            optimizer.zero_grad()
            
            recon = model(batch)      # shape: (B, T, F)
            loss = criterion(recon, batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch.size(0)
        
        train_loss = train_loss_sum / len(train_loader.dataset)
        
        # -------- Validation --------
        model.eval()
        valid_loss_sum = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                valid_loss_sum += loss.item() * batch.size(0)
        
        valid_loss = valid_loss_sum / len(valid_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")
        
        # モデルの保存
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_lstm_autoencoder-5-2.pth")
            print("  -> best model saved.")

# -----------------------------------------------------
# 4) メイン関数: データ準備＆学習の流れ
# -----------------------------------------------------
def main():
    train_dir = "./data/pitch-train"
    valid_dir = "./data/pitch-valid"
    
    train_files = glob.glob(os.path.join(train_dir, "*.wav"))
    valid_files = glob.glob(os.path.join(valid_dir, "*.wav"))
    print(f"訓練データ: {len(train_files)} ファイル, 検証データ: {len(valid_files)} ファイル")
    
    # Dataset & DataLoader
    train_dataset = AudioDataset(train_files)
    valid_dataset = AudioDataset(valid_files)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # LSTM Autoencoder インスタンス作成
    # 入力次元(input_dim)=128, hidden_dim=64, latent_dim=32, num_layers=1 など
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64, latent_dim=32, num_layers=1)
    model.to(device)
    
    # 学習
    train_lstm_autoencoder(
        model, train_loader, valid_loader, 
        device=device, 
        num_epochs=100, 
        lr=1e-3
    )
    
    print("学習完了！")

if __name__ == '__main__':
    main()
