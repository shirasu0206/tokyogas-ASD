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

class AudioAutoencoder(nn.Module):
    """ダミーの AudioAutoencoder (推論には利用しない)"""
    def __init__(self, *args, **kwargs):
        super(AudioAutoencoder, self).__init__()
    def forward(self, x):
        return x

def plot_reconstruction_errors(file_errors, threshold):
    # ファイル名リストとエラー値リストを作成
    file_names = [os.path.basename(f) for f in file_errors.keys()]
    errors = list(file_errors.values())
    
    # ファイル名の先頭文字に基づき色とグループを割り当てる（元のルール）
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
    
    # 表示順序は「その他(青)」(group0) → 「緑」(group1) → 「黄色」(group2) → 「紫」(group3)
    items_sorted = sorted(items, key=lambda x: x[0])
    sorted_file_names = [item[1] for item in items_sorted]
    sorted_errors = [item[2] for item in items_sorted]
    sorted_colors = [item[3] for item in items_sorted]
    
    # 並び替えた後、ファイル名に日本語が含まれている場合は色を赤に上書きする
    final_colors = []
    for name, color in zip(sorted_file_names, sorted_colors):
        if contains_japanese(name):
            final_colors.append('red')
        else:
            final_colors.append(color)

    plt.figure(figsize=(12, 6))
    # x軸はインデックス、ファイル名は表示しない
    plt.bar(range(len(sorted_errors)), sorted_errors, color=final_colors, label='Error (Negative Log-likelihood)')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.6f})")
    plt.xlabel("File Index")
    plt.ylabel("Error (Higher -> More Anomalous)")
    plt.title("Anomaly Score for Each File (GMM based on Negative Log-likelihood)")
    plt.xticks([])  # x軸ラベルは非表示
    plt.legend()
    plt.tight_layout()
    
    plt.yscale('log')

    output_path = "reconstruction_error.png"
    plt.savefig(output_path)
    plt.show()
    
    print(f"✅ 異常度スコアのグラフを {output_path} に保存しました！")


def test_model():
    start_time = time.time()

    # 1) 学習済み GMM, Scaler, PCA をロード
    gmm = joblib.load("gmm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    print("GMM, Scaler, PCA をロードしました。")

    model = AudioAutoencoder()

    test_dir = "./data/noised-pitched-test"
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

        flat_scaled = scaler.transform(flat)
        flat_pca = pca.transform(flat_scaled)
        log_likelihood = gmm.score_samples(flat_pca)  # shape=(1,)
        error = -log_likelihood[0]
        file_errors[file_path[0]] = error

    # 3) 統計量からしきい値を決定
    errors = np.array(list(file_errors.values()))
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    threshold = mean_err + 2 * std_err  # 例: 平均+2σ
    threshold = 220  # 固定値を使う場合

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
