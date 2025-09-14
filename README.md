## tokyogas-ASD リポジトリ – 詳細 README
# 概要

このリポジトリは、東京ガスが保有する「ASD（Anomaly Sound Detection）」の検証用コードとデータ構造です。主な目的は、音響データから異常音を検出するための機械学習モデルを構築・評価することです。CNN‑ベースのオートエンコーダ、LSTMオートエンコーダ、ガウス混合モデル(GMM)＋One‑Class SVM など複数のアプローチが実装されています。動画ファイルを音声に変換するためのスクリプトや、元データにピッチ変更やノイズ付与を行う前処理スクリプトも含まれます。

# ディレクトリ構成

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
