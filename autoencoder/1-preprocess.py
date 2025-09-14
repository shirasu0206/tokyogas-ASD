import os
import subprocess
import glob
import random
import shutil  # 一時ディレクトリの削除用

# 設定
INPUT_DIR = "./data/input-mp3data"
TRAIN_DIR = "./data/train"
VALID_DIR = "./data/valid"
TEST_DIR = "./data/test"
SEGMENT_LENGTH_SEC = 10  # 分割サイズ（秒）
TRAIN_RATIO = 0.72        # Train に振り分ける割合
VALID_RATIO = 0.18       # Valid に振り分ける割合（Test も 15%）

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 出力フォルダ作成
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# MP3ファイルの取得
mp3_files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
if not mp3_files:
    print(f"入力フォルダ {INPUT_DIR} にMP3が見つかりません。")
    exit()

for file_path in mp3_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {file_path} ...")

    # 各ファイルごとに一時的な出力フォルダを作成
    temp_output_dir = os.path.join(INPUT_DIR, "temp_segments", base_name)
    os.makedirs(temp_output_dir, exist_ok=True)

    # 出力ファイル名のパターン（各MP3ごとに固有の名前を付与）
    temp_output_pattern = os.path.join(temp_output_dir, f"{base_name}_seg%04d.wav")

    # ffmpeg コマンド:
    # - 入力 MP3 を指定秒数ごとに分割し、
    # - PCM 16-bit little-endian、16kHz、モノラルに再エンコードして出力
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", file_path,                     # 入力MP3
        "-f", "segment",                     # セグメント処理
        "-segment_time", str(SEGMENT_LENGTH_SEC),  # 指定秒数ごとに分割
        "-c:a", "pcm_s16le",                 # PCM 16-bit little-endian にエンコード
        "-ar", "16000",                      # サンプルレート 16kHz
        "-ac", "1",                          # モノラル
        temp_output_pattern
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"ffmpegエラー: {e}")
        continue

    # 分割されたファイルのリストを取得
    wav_files = sorted(glob.glob(os.path.join(temp_output_dir, f"{base_name}_seg*.wav")))

    # 各ファイルをランダムに Train/Valid/Test に振り分け
    for wav_file in wav_files:
        rand_val = random.random()
        if rand_val < TRAIN_RATIO:
            target_dir = TRAIN_DIR
        elif rand_val < TRAIN_RATIO + VALID_RATIO:
            target_dir = VALID_DIR
        else:
            target_dir = TEST_DIR

        os.rename(wav_file, os.path.join(target_dir, os.path.basename(wav_file)))

    print(f"  {len(wav_files)} 個のセグメントを作成しました。")

    # 一時ディレクトリを再帰的に削除
    try:
        shutil.rmtree(temp_output_dir)
    except Exception as e:
        print(f"一時ディレクトリ削除エラー: {e}")

print("全処理完了！")
