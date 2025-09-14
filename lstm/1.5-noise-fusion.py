from pydub import AudioSegment
import random
import os

# メインの音声ファイルが入っているフォルダ
MAIN_DIR = "./data/test"
# 重ねたい自然音（ノイズ）が入っているフォルダ
NOISE_DIR = "./data/noise"
# 出力先フォルダ
OUTPUT_DIR = "./data/noised-test"
# ノイズを追加する音声の数
NUM_NOISE_ADDED = 30  # 例: 5個だけランダムにノイズを追加する

def mix_random_noise(main_audio_path, noise_audio_path, output_path):
    """
    main_audio_path で指定した音声に
    noise_audio_path で指定した自然音をランダムな位置・音量で重ねて
    output_path に書き出す
    """
    # メイン音声を読み込む
    main_audio = AudioSegment.from_file(main_audio_path)

    # ノイズ音を読み込む
    noise_audio = AudioSegment.from_file(noise_audio_path)

    # ノイズ音を重ねる位置（ミリ秒単位で計算）
    if len(noise_audio) < len(main_audio):
        max_offset = len(main_audio) - len(noise_audio)
        offset = random.randint(0, max_offset)
    else:
        offset = 0  # ノイズ音が長すぎる場合は、先頭に挿入（切り詰めることも可能）

    # ノイズの音量をランダムに変化 (-10dB ~ +15dB の範囲)
    noise_dB_change = round(random.uniform(-10.0, 10.0), 1)  # 小数第1位まで表示
    noise_audio = noise_audio + noise_dB_change

    # overlay を使って音を重ね合わせる
    mixed_audio = main_audio.overlay(noise_audio, position=offset)

    # 書き出し（フォーマットは wav）
    mixed_audio.export(output_path, format="wav")

    # 追加したノイズ情報を返す
    return noise_dB_change

def main():
    # フォルダ内の音声ファイルを取得
    main_files = [f for f in os.listdir(MAIN_DIR) if f.endswith(".wav")]
    noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith(".m4a")]

    # 出力フォルダを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ノイズを付与する音声をランダムに選択（NUM_NOISE_ADDED 個）
    if NUM_NOISE_ADDED > len(main_files):
        selected_files = main_files  # もし音声ファイル数よりNUM_NOISE_ADDEDが多かったら全部対象
    else:
        selected_files = random.sample(main_files, NUM_NOISE_ADDED)

    # 音声を処理
    for main_file in main_files:
        main_path = os.path.join(MAIN_DIR, main_file)
        
        # ファイル名変更のため、拡張子を分離
        file_name, file_ext = os.path.splitext(main_file)

        if main_file in selected_files:
            # ランダムにノイズファイルを1つ選択
            noise_file = random.choice(noise_files)
            noise_path = os.path.join(NOISE_DIR, noise_file)

            # ノイズの種類（拡張子を除くファイル名）
            noise_name, _ = os.path.splitext(noise_file)

            # ノイズを追加し、音量変更値を取得
            noise_dB_change = mix_random_noise(main_path, noise_path, f"temp{file_ext}")

            # 新しいファイル名を作成（例: audio_01_noisy_bang_+5dB.wav）
            output_file_name = f"{file_name}_noisy_{noise_name}_{'+' if noise_dB_change >= 0 else ''}{noise_dB_change}dB{file_ext}"
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            # ファイルをリネーム
            os.rename("temp.wav", output_path)

            print(f"✅ Processed {main_file} with noise {noise_file} ({noise_dB_change}dB) -> {output_file_name}")
        else:
            # ノイズを追加しない場合はそのままコピー（名前は変更しない）
            output_path = os.path.join(OUTPUT_DIR, main_file)
            original_audio = AudioSegment.from_file(main_path)
            original_audio.export(output_path, format="wav")
            print(f"➡️ Copied {main_file} without noise")

if __name__ == "__main__":
    main()
