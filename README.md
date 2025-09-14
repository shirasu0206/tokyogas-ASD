# tokyogas-ASD リポジトリ – 詳細 README
## 概要

このリポジトリは、東京ガス様とのプロジェクトの一環である「ASD（Anomaly Sound Detection）」の検証用コードとデータ構造です。主な目的は、音響データから異常音を検出するための機械学習モデルを構築・評価することです。CNN‑ベースのオートエンコーダ、LSTMオートエンコーダ、ガウス混合モデル(GMM)＋One‑Class SVM など複数のアプローチが実装されています。動画ファイルを音声に変換するためのスクリプトや、元データにピッチ変更やノイズ付与を行う前処理スクリプトも含まれます。

## ディレクトリ構成

下表は本リポジトリの主要なディレクトリ／ファイルと、その役割の概要です。

| パス                                                  | 概要                                                                                                                                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mp4tomp3/change.py`                                | MP4 動画から音声（MP3）を抽出するスクリプト。MP4 を `mp4tomp3/input` に置き、実行すると MP3 が `mp4tomp3/output` に保存されます。                                                                 |
| `autoencoder/1-preprocess.py`                       | 音声データ前処理。`./data/input-mp3data` 配下の MP3 を 10 秒ごとの WAV に分割し、ランダムに train/valid/test ディレクトリへ振り分けます。ffmpeg を使用して 16kHz・モノラルに変換しています。                            |
| `autoencoder/1.3-change-pitch.py`                   | テスト用 WAV に対して半音単位でピッチシフトし、`./data/pitch-test` に書き出します。半音のずれは `PITCH_SHIFT_VALUES` で指定します。                                                                   |
| `autoencoder/1.5-noise-fusion.py`                   | 自然環境音をランダムな位置・音量で重ね、ノイズ付きデータを生成します。メインの音声 (`./data/test`) とノイズ音 (`./data/noise`) を読み込み、ミックス結果を `./data/noised-test` に保存します。                                 |
| `autoencoder/2-train-autoencoder.py`                | CNN 型オートエンコーダの学習。`./data/pitch-train` と `./data/pitch-valid` の WAV からメルスペクトログラムを作成し、モデルを 100 エポック学習します。最良モデルは `best_autoencoder.pth` などとして保存されます。           |
| `autoencoder/3-test-autoencoder.py`                 | 学習済みモデルでテストデータ（`./data/test`）を評価し、再構成誤差から異常を検出します。閾値を越えたファイルを `./data/anomalies` へコピーし、評価指標も出力します。                                                          |
| `autoencoder/3.1-test-pitch.py`                     | ノイズやピッチ変更を施したテストデータ (`./data/noised-pitched-test`) 用の評価スクリプト。閾値の設定や評価方法は 3-test-autoencoder.py と類似ですが、重みファイルやテストディレクトリが異なります。                               |
| `lstm/2-train-lstm.py`                              | LSTM オートエンコーダの学習。メルスペクトログラムの時間方向を系列データとして扱い、Encoder/Decoder に LSTM を用います。データ読み込みと学習の流れは autoencoder と同様。                                                    |
| `lstm/3-test-lstm.py`                               | 学習済み LSTM オートエンコーダで評価を行うスクリプト。メルスペクトログラムを (time, feature) の形に変換して推論し、閾値に基づいて異常を判断します。                                                                       |
| `oneclass-svm/2-train-svm.py`                       | One‑Class SVM の前段として GMM を用いたモデル学習。メルスペクトログラムをベクトルに展開し、StandardScaler で正規化した後に PCA で次元削減し、GaussianMixture で学習します。学習済みモデルと scaler/pca は `.pkl` ファイルとして保存します。 |
| `oneclass-svm/3-test-svm.py` / `3-1-pitich-test.py` | 学習済み GMM・PCA・scaler を読み込み、負の対数尤度を異常スコアとして算出します。閾値を超えたファイルを `./data/anomalies` に保存し、スコアのグラフを出力します。                                                           |
| `data1/noise` / `data5/noise`                       | サンプルの自然音（机を叩く音・火の音など）を収録したフォルダ。ノイズ付与スクリプトで使用します。                                                                                                            |
| `.gitignore`                                        | 音声データフォルダ (`autoencoder/data`, `data1/input-mp3data` など) は Git 管理の対象外とする設定。                                                                                 |

## 前提環境

Python 3.8 以降。

ffmpeg : MP4 → MP3 変換や WAV 切り出しに使用します。各スクリプトは ffmpeg コマンドを直接呼び出しているため、パスが通っている必要があります。

pip install numpy torch librosa pydub scikit-learn joblib matplotlib など。LSTM/Autoencoder では PyTorch を用い、One‑Class SVM では scikit‑learn を使用します。

pydub を利用する際は ffmpeg が必要です（環境変数 FFMPEG_BINARY でパスを指定するか、システムにインストールしてください）。

GPU (CUDA) があれば学習が高速になりますが、無くても動作します。

## 使用方法

以下に、動画→音声抽出からデータ生成、学習、評価までの一例を紹介します。各ステップに応じて適切なディレクトリへファイルを置き換えて下さい。

### 1. 動画から音声の抽出 (任意)

MP4 ファイルを mp4tomp3/input に配置します。

python mp4tomp3/change.py を実行すると、各動画から音声が抽出され、mp4tomp3/output に MP3 ファイルが保存されます。出力された MP3 を次のステップで使用する ./data/input-mp3data にコピーします。

### 2. MP3 から WAV 分割・データセット生成

data/input-mp3data ディレクトリに MP3 ファイル（または上記で生成した MP3）を保存します。

python autoencoder/1-preprocess.py（または lstm/1-preprocess.py, oneclass-svm/1-preprocess.py）を実行すると、MP3 が 10 秒単位の WAV に変換され、ランダムに学習用(./data/train)、検証用(./data/valid)、テスト用(./data/test)に振り分けられます。

SEGMENT_LENGTH_SEC や分割比率はスクリプト内の定数で変更できます。

処理後の一時フォルダは自動で削除されます。

### 3. データ拡張（ピッチ変更・ノイズ付与）

ピッチ変更: python autoencoder/1.3-change-pitch.py を実行すると、./data/test 内の WAV に対して指定した半音でピッチシフトしたデータを生成し、./data/pitch-test に保存します
。学習用のデータに対してピッチ変更したい場合は MAIN_DIR を ./data/train に書き換えて実行し、結果を ./data/pitch-train へコピーします。

ノイズ付与: python autoencoder/1.5-noise-fusion.py（または lstm/1.5-noise-fusion.py, oneclass-svm/1.5-noise-fusion.py）を実行すると、MAIN_DIR で指定した音声に対して、data/noise ディレクトリ内の自然音をランダムに重ねて OUTPUT_DIR に保存します。
ノイズを追加しない場合は元のファイルがそのままコピーされます。

ピッチ変更とノイズ付与を組み合わせたい場合は、ピッチ変更で生成したファイルに対してノイズ付与を再度実行し、./data/noised-pitched-test や ./data/noised-pitched-train を作成します。

### 4. モデルの学習
#### Autoencoder（CNN）

前処理・拡張後のデータを ./data/pitch-train（学習）、./data/pitch-valid（検証）に配置します。

python autoencoder/2-train-autoencoder.py を実行すると、Mel スペクトログラムを入力とする CNN オートエンコーダが学習されます。学習過程では訓練損失と検証損失が表示され、検証損失が最小になったモデルが best_autoencoder.pth や best_autoencoder-5-2.pth などとして保存されます。

#### LSTM オートエンコーダ

同様に ./data/pitch-train／./data/pitch-valid に音声データを準備します。

python lstm/2-train-lstm.py を実行すると、メルスペクトログラムを系列データとして扱う LSTM オートエンコーダが学習されます。学習が進むと最良モデルが best_lstm_autoencoder-5-2.pth などに保存されます。

#### GMM ＋ One‑Class SVM

data/pitch-train に学習用 WAV を準備します。

python oneclass-svm/2-train-svm.py を実行すると、Mel スペクトログラムをフラットな特徴ベクトルに展開し、StandardScaler で正規化し、PCA で次元削減した後、GaussianMixture により異常度モデルを学習します。学習済みモデルは gmm_model.pkl、scaler.pkl、pca.pkl として保存されます。

### 5. 評価・異常検出
#### Autoencoder 系

通常テスト：python autoencoder/3-test-autoencoder.py を実行すると、./data/test の WAV に対して再構成誤差を計算し、閾値を超えるファイルを異常と判断します。異常と判定された音声は ./data/anomalies にコピーされ、TP/FP/TN/FN などの評価指標が表示されます。

ピッチ・ノイズ付きテスト：python autoencoder/3.1-test-pitch.py では ./data/noised-pitched-test のデータに対して同様の処理を行います。閾値はスクリプト内で threshold = 0.0012 などに設定されており、必要に応じて調整します。

#### LSTM 系

python lstm/3-test-lstm.py を実行します。LSTM オートエンコーダの出力との再構成誤差を計算し、閾値を超えるデータを異常とします。結果のグラフは reconstruction_error.png として保存されます。

#### One‑Class SVM (GMM)

python oneclass-svm/3-test-svm.py は、ノイズ付きデータ(./data/noised-test)に対して GMM の対数尤度を計算し、負の対数尤度が閾値より大きいものを異常と判定します。
閾値は平均＋2σに基づいて計算後、固定値（例: 150）に変更して利用する例が示されています。

python oneclass-svm/3-1-pitich-test.py はピッチ付き・ノイズ付きのデータセットに対する評価版です。処理の流れは 3-test-svm.py と同様です。

### データ配置ガイド

#### MP4 動画: 
mp4tomp3/input に入れ、change.py で音声を抽出して mp4tomp3/output へ。抽出後の MP3 は data/input-mp3data へコピーします。ここでいうmp4動画とは東京ガス様から頂いた現場データです。google drive からダウンロードしてください。

#### 学習／検証／テスト音声 (WAV): 
1-preprocess.py で自動生成されます。TRAIN_DIR、VALID_DIR、TEST_DIR はスクリプト内で ./data/train、./data/valid、./data/test に設定されています。

ノイズ音: data/noise や data1/noise に m4a 形式で配置します。リポジトリには机や火の音などサンプルが含まれています。
自身で収集したノイズもここに追加できます。

#### ピッチ変更結果: 
1.3-change-pitch.py で生成した WAV を data/pitch-test または data/pitch-train 等に保存します。PITCH_SHIFT_VALUES を変更すればシフト量を変えられます。

#### ノイズ付与結果: 
1.5-noise-fusion.py 実行後、結果は data/noised-test 等に保存されます。テストだけノイズを付加する場合は NUM_NOISE_ADDED を変更して対象ファイル数を調整します。

#### 学習済みモデル: 
各トレーニングスクリプト実行後、.pth や .pkl のファイルが保存されます。テストスクリプト内でファイル名を指定してロードしています（best_autoencoder-5-1.pth など）。
必要に応じてファイル名を書き換えてください。
