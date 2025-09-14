import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import shutil
import matplotlib.pyplot as plt
import re
import torch.nn as nn
import joblib  # gmm_model, scaler, pca の読み込みに利用
import time

# ---------------------------
# ファイル名に日本語が含まれているかチェックする関数
# ---------------------------
def contains_japanese(text):
    return bool(re.search("[\u3040-\u30FF\u4E00-\u9FFF]", text))

# ---------------------------
# TestAudioDataset の定義
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
            mel_spec_norm = np.pad(
                mel_spec_norm, ((0, 0), (0, pad_width)), 
                mode='constant'
            )
        elif current_width > self.desired_width:
            mel_spec_norm = mel_spec_norm[:, :self.desired_width]

        # 転置して (311, 128) にする
        mel_spec_norm = mel_spec_norm.T
        
        tensor = torch.tensor(mel_spec_norm, dtype=torch.float32)
        return tensor, file_path

# ---------------------------
# ダミーの AudioAutoencoder (推論には利用しない)
# ---------------------------
class AudioAutoencoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AudioAutoencoder, self).__init__()
    def forward(self, x):
        return x

# ---------------------------
# 再構成誤差のグラフを作成・保存 (x軸はインデックス表示)
# ---------------------------
def plot_reconstruction_errors(file_errors, threshold):
    # ファイル名リスト（ground truth: 日本語が含まれているかどうかで判定）
    file_names = [os.path.basename(f) for f in file_errors.keys()]
    errors = list(file_errors.values())
    
    # ファイル名に日本語が含まれている場合は赤、含まれていなければ青
    bar_colors = ['red' if contains_japanese(name) else 'blue' for name in file_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(errors)), errors, color=bar_colors, label='Error (Negative Log-likelihood)')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.6f})")
    plt.xlabel("File Index")
    plt.ylabel("Error (Higher -> More Anomalous)")
    plt.title("Anomaly Score for Each File (GMM based on Negative Log-likelihood)")
    plt.xticks([])  # ファイル名のラベルは非表示
    plt.legend()
    
    # 縦軸を対数スケールに設定
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = "reconstruction_error.png"
    plt.savefig(output_path)
    plt.show()
    
    print(f"✅ 異常度スコアのグラフを {output_path} に保存しました！")


# ---------------------------
# メイン処理: 異常検出および評価指標の計算
# ---------------------------
def test_model():
    start_time = time.time()

    # 1) 学習済み GMM, Scaler, PCA をロード
    gmm = joblib.load("gmm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    print("GMM, Scaler, PCA をロードしました。")

    model = AudioAutoencoder()

    test_dir = "./data/noised-test"
    anomaly_dir = "./data/anomalies"
    os.makedirs(anomaly_dir, exist_ok=True)
    
    test_files = glob.glob(os.path.join(test_dir, "*.wav"))
    print(f"検証データ: {len(test_files)} ファイル")
    
    test_dataset = TestAudioDataset(test_files, sr=16000, n_mels=128, desired_width=311)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    file_errors = {}

    # 2) 推論（GMMによる対数尤度計算 → 異常度）
    for tensor, file_path in test_loader:
        # tensor shape: (1, 311, 128)
        arr_3d = tensor.numpy()      # shape=(1, 311, 128)
        flat = arr_3d.reshape(1, -1)   # shape=(1, 39808)

        # scaler transform
        flat_scaled = scaler.transform(flat)
        # pca transform
        flat_pca = pca.transform(flat_scaled)
        # gmm.score_samples → 対数尤度 (大きいほど正常)
        log_likelihood = gmm.score_samples(flat_pca)  # shape=(1,)
        # 異常度エラー = -log_likelihood
        error = -log_likelihood[0]
        file_errors[file_path[0]] = error

    # 3) 統計量からしきい値を決定し、異常ファイルを抽出
    errors = np.array(list(file_errors.values()))
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    threshold = mean_err + 2 * std_err  # 例: 平均+2σ
    threshold = 150  # 固定値を使う場合

    print(f"異常スコアの平均: {mean_err:.6f}, 標準偏差: {std_err:.6f}")
    print(f"異常検出の閾値: {threshold:.6f}")

    anomalies = []
    for fpath, err in file_errors.items():
        if err > threshold:
            anomalies.append((fpath, err))
            base_name = os.path.basename(fpath)
            dest_path = os.path.join(anomaly_dir, base_name)
            shutil.copy(fpath, dest_path)
            print(f"異常検出: {base_name} | スコア: {err:.6f}")
    
    print(f"合計 {len(anomalies)} 個の異常ファイルを検出・保存しました。")

    # 4) 評価指標の計算 (TP, FP, TN, FN)
    # 正解ラベルは「ファイル名に日本語が含まれている場合」を異常 (ground truth) とする
    TP = FP = TN = FN = 0
    for fpath, err in file_errors.items():
        base_name = os.path.basename(fpath)
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

    # 5) エラーをグラフ化
    plot_reconstruction_errors(file_errors, threshold)

    end_time = time.time()
    print(f"処理完了 (経過時間: {end_time - start_time:.2f} 秒)")

if __name__ == '__main__':
    test_model()
