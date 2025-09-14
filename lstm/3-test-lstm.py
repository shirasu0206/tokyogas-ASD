import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shutil
import matplotlib.pyplot as plt
import re
import time

# ---------------------------
# ファイル名に日本語が含まれているかチェックする関数
# ---------------------------
def contains_japanese(text):
    return bool(re.search("[\u3040-\u30FF\u4E00-\u9FFF]", text))

# ---------------------------
# TestAudioDataset の定義 (変更なし)
# ---------------------------
class TestAudioDataset(Dataset):
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
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            n_fft=1024, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = np.clip((mel_spec_db + 80) / 80, 0, 1)

        current_width = mel_spec_norm.shape[1]
        if current_width < self.desired_width:
            pad_width = self.desired_width - current_width
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        elif current_width > self.desired_width:
            mel_spec_norm = mel_spec_norm[:, :self.desired_width]

        # (128, 311) → (1, 128, 311) としてチャネル次元を追加
        tensor = np.expand_dims(mel_spec_norm, axis=0)
        tensor = torch.tensor(tensor, dtype=torch.float32)
        
        return tensor, file_path

# ---------------------------
# LSTMベースのオートエンコーダ (AudioAutoencoder)
# ---------------------------
class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=32, num_layers=1):
        """
        input_dim : LSTMの特徴次元 (メル周波数数=128)
        hidden_dim: LSTMの隠れ状態次元
        latent_dim: 潜在ベクトルの次元
        num_layers: LSTMのスタック数
        """
        super(AudioAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # ---------------- Encoder LSTM ----------------
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,  # 入力を (B, T, F) とみなす
            bidirectional=False
        )
        self.enc_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        # ---------------- Decoder LSTM ----------------
        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.dec_output = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        """
        x: shape = (B, 1, 128, 311)
           (B=バッチ, 1=チャネル, 128=メル周波数, 311=時間フレーム)
        """
        # 1) チャネル次元を削除 → (B, 128, 311)
        x = x.squeeze(1)
        # 2) 転置して (B, T, F) にする → (B, 311, 128)
        x = x.transpose(1, 2)

        # ---------- Encoder ----------
        out, (h_n, c_n) = self.encoder_lstm(x)  # out: (B, T, hidden_dim)
        h_enc = h_n[-1]  # 最上位層の hidden state, shape: (B, hidden_dim)
        z = self.enc_linear(h_enc)  # (B, latent_dim)

        # ---------- Decoder ----------
        B, T, F = x.size()  # T=311, F=128
        z_repeated = z.unsqueeze(1).repeat(1, T, 1)  # (B, 311, latent_dim)
        dec_out, (dec_h, dec_c) = self.decoder_lstm(z_repeated)  # (B, 311, hidden_dim)
        x_recon = self.dec_output(dec_out)  # (B, 311, 128)

        # 3) 逆転置して元の形に戻す → (B, 128, 311)
        x_recon = x_recon.transpose(1, 2)
        # 4) チャネル次元を追加 → (B, 1, 128, 311)
        x_recon = x_recon.unsqueeze(1)

        return x_recon

# ---------------------------
# 再構成誤差のグラフを作成・保存（並び替え後に日本語が含まれている場合は赤に上書き）
# ---------------------------
def plot_reconstruction_errors(file_errors, threshold):
    # ファイル名リストとエラー値リストを作成
    file_names = [os.path.basename(f) for f in file_errors.keys()]
    errors = list(file_errors.values())
    
    # 初期の色とグループの割り当て
    def get_color_and_group(name):
        if name and name[0] == '6':
            return 'purple', 3
        elif name and name[0] == '3':
            return 'yellow', 2
        elif name and name[0] == '1':
            return 'green', 1
        else:
            return 'blue', 0

    items = []
    for name, error in zip(file_names, errors):
        color, group = get_color_and_group(name)
        items.append((group, name, error, color))
    
    # 並び替え: グループ0（その他＝青） → グループ1（緑） → グループ2（黄色） → グループ3（紫）
    items_sorted = sorted(items, key=lambda x: x[0])
    sorted_file_names = [item[1] for item in items_sorted]
    sorted_errors = [item[2] for item in items_sorted]
    sorted_colors = [item[3] for item in items_sorted]
    
    # 並び替えた後、もしファイル名に日本語が含まれていれば色を赤に上書き
    final_colors = []
    for name, color in zip(sorted_file_names, sorted_colors):
        if contains_japanese(name):
            final_colors.append('red')
        else:
            final_colors.append(color)
    
    plt.figure(figsize=(12, 6))
    # x軸はインデックス、ファイル名ラベルは非表示
    plt.bar(range(len(sorted_errors)), sorted_errors, color=final_colors, label='Reconstruction Error')
    plt.axhline(y=threshold, color='black', linestyle='--', label=f"Threshold ({threshold:.6f})")
    plt.xlabel("File Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error for Each File")
    plt.xticks([])  # x軸ラベルは非表示
    plt.legend()
    plt.tight_layout()

    output_path = "reconstruction_error.png"
    plt.savefig(output_path)
    plt.show()
    
    print(f"✅ 再構成誤差のグラフを {output_path} に保存しました！")

# ---------------------------
# テスト＆異常検出のメイン処理
# ---------------------------
def test_model():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dir = "./data/noised-pitched-test"
    anomaly_dir = "./data/anomalies"
    os.makedirs(anomaly_dir, exist_ok=True)
    
    test_files = glob.glob(os.path.join(test_dir, "*.wav"))
    print(f"検証データ: {len(test_files)} ファイル")
    
    test_dataset = TestAudioDataset(test_files, sr=16000, n_mels=128, desired_width=311)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # --- LSTM オートエンコーダを使用 ---
    model = AudioAutoencoder(
        input_dim=128, 
        hidden_dim=64, 
        latent_dim=32, 
        num_layers=1
    ).to(device)

    model.load_state_dict(torch.load("best_lstm_autoencoder-5-2.pth", map_location=device))
    model.eval()
    
    errors = []
    file_errors = {}

    with torch.no_grad():
        for tensor, file_path in test_loader:
            tensor = tensor.to(device)  # (B=1, 1, 128, 311)
            output = model(tensor)       # (1, 1, 128, 311)
            error = ((tensor - output) ** 2).mean().item()
            errors.append(error)
            file_errors[file_path[0]] = error
    
    errors = np.array(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    # しきい値は平均+2σ（例として固定値も設定可能）
    threshold = mean_err + 2 * std_err
    threshold = 0.0025  # 必要に応じて調整してください
    print(f"再構成誤差の平均: {mean_err:.6f}, 標準偏差: {std_err:.6f}")
    print(f"異常検出の閾値: {threshold:.6f}")
    
    anomalies = []
    for file_path, err in file_errors.items():
        if err > threshold:
            anomalies.append((file_path, err))
            base_name = os.path.basename(file_path)
            dest_path = os.path.join(anomaly_dir, base_name)
            shutil.copy(file_path, dest_path)
            print(f"異常検出: {base_name} | 再構成誤差: {err:.6f}")
    
    print(f"合計 {len(anomalies)} 個の異常ファイルを検出・保存しました。")
    
    # ---------------------------
    # 評価指標の計算 (TP, FP, TN, FN)
    # ---------------------------
    # 正解ラベルは「ファイル名に日本語が含まれている場合」とする
    TP = FP = TN = FN = 0
    for file_path, err in file_errors.items():
        base_name = os.path.basename(file_path)
        is_ground_truth_anomaly = contains_japanese(base_name)
        is_detected_anomaly = err > threshold
        
        if is_detected_anomaly and is_ground_truth_anomaly:
            TP += 1
        elif is_detected_anomaly and not is_ground_truth_anomaly:
            FP += 1
        elif (not is_detected_anomaly) and (not is_ground_truth_anomaly):
            TN += 1
        elif (not is_detected_anomaly) and is_ground_truth_anomaly:
            FN += 1

    print("\n=== 評価指標 ===")
    print(f"真陽性 (TP): {TP}")
    print(f"偽陽性 (FP): {FP}")
    print(f"真陰性 (TN): {TN}")
    print(f"偽陰性 (FN): {FN}")

    plot_reconstruction_errors(file_errors, threshold)
    
    end_time = time.time()
    print(f"処理完了 (経過時間: {end_time - start_time:.2f} 秒)")

if __name__ == '__main__':
    test_model()
