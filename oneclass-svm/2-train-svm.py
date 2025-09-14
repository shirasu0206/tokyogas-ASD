import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture  # ★ GMMを追加
import joblib  # モデルの保存/読み込みに利用

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

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            n_fft=1024, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # [-80, 0] → [0, 1] にクリップ正規化
        mel_spec_norm = np.clip((mel_spec_db + 80) / 80, 0, 1)

        current_width = mel_spec_norm.shape[1]
        if current_width < self.desired_width:
            pad_width = self.desired_width - current_width
            mel_spec_norm = np.pad(
                mel_spec_norm, 
                pad_width=((0, 0), (0, pad_width)), 
                mode='constant'
            )
        else:
            mel_spec_norm = mel_spec_norm[:, :self.desired_width]
        
        # shape: (128, 311) → 転置 (311, 128)
        mel_spec_norm = mel_spec_norm.T  
        
        return torch.tensor(mel_spec_norm, dtype=torch.float32), file_path


def main():
    train_dir = "./data/pitch-train"
    train_files = glob.glob(os.path.join(train_dir, "*.wav"))
    print(f"訓練データ: {len(train_files)} ファイル")

    # ---- Dataset作成 ----
    train_dataset = AudioDataset(train_files)
    
    # ---- 特徴量をまとめて取得 ----
    train_features = []
    for i in range(len(train_dataset)):
        mel_2d, file_path = train_dataset[i]   # shape = (311, 128)
        vec = mel_2d.numpy().flatten()         # shape: (311*128=39808,)
        train_features.append(vec)

    train_features = np.stack(train_features, axis=0)
    print("train_features shape:", train_features.shape)  # (N, 39808)
    
    # ---- スケーリング (StandardScaler) ----
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    print("Scaler fit 完了。shape:", train_features_scaled.shape)

    # ---- PCA で次元削減 (例: n_components=100) ----
    pca = PCA(n_components=100)
    train_features_pca = pca.fit_transform(train_features_scaled)
    print("PCA fit 完了。shape:", train_features_pca.shape)

    # ---- Gaussian Mixture Model (GMM) で学習 ----
    # ここでは例として n_components=5, covariance_type='full' を使用
    gmm = GaussianMixture(
        n_components=5,
        covariance_type='full',
        random_state=42
    )
    print("GMM で学習を開始...")
    gmm.fit(train_features_pca)
    print("...学習完了！")

    # ---- モデルおよび scaler, pca を保存 ----
    joblib.dump(gmm, "gmm_model.pkl")     # GMMモデル
    joblib.dump(scaler, "scaler.pkl")     # Scaler
    joblib.dump(pca, "pca.pkl")           # PCA
    print("✅ 学習済みモデル(gmm_model.pkl), scaler(scaler.pkl), pca(pca.pkl) を保存しました。")


if __name__ == '__main__':
    main()
