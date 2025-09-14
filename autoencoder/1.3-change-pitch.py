import os
from pydub import AudioSegment

# メインの音声ファイルが入っているフォルダ
MAIN_DIR = "./data/test"
# 出力先フォルダ
OUTPUT_DIR = "./data/pitch-test"
# ピッチを変化させたい半音のリスト（５段階の例）
PITCH_SHIFT_VALUES = [-6, -3, -1, 0, 1, 3, 6]

def pitch_shift(sound, semitones):
    """
    sound(AudioSegment) を semitones 半音だけピッチシフトする
    ピッチシフトはサンプリングレートを変更して行い、
    その後、元のフレームレートに戻す方法を使います。
    """
    # 1半音(セミトーン)ごとの周波数比は 2^(1/12)
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    
    # _spawn() で raw_data を使い回しつつ新しいサンプリングレートを指定
    shifted_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    
    # 見かけ上の再生速度は変わるが、最終的には元のフレームレートに戻す
    return shifted_sound.set_frame_rate(sound.frame_rate)

def process_audio_with_pitch_shift(input_path, output_dir, pitch_shift_values):
    """
    input_path で指定された音声ファイルを読み込み、
    pitch_shift_values で指定された各半音分だけ
    ピッチシフトしたファイルを作成し、output_dir に保存する
    """
    # 音声読み込み
    original_audio = AudioSegment.from_file(input_path)
    
    # 入力ファイル名と拡張子を分解
    base_name = os.path.basename(input_path)
    file_name, ext = os.path.splitext(base_name)

    for semitones in pitch_shift_values:
        if semitones == 0:
            continue  # ピッチ変更なしの処理はスキップ

        shifted_audio = pitch_shift(original_audio, semitones)
        
        shift_prefix = f"{abs(semitones)}{'+' if semitones > 0 else '-' if semitones < 0 else ''}"
        output_file_name = f"{shift_prefix}_{file_name}{ext}"
        output_path = os.path.join(output_dir, output_file_name)
        
        shifted_audio.export(output_path, format="wav")
        print(f"✅ {base_name} -> {output_file_name} (ピッチ {shift_prefix}半音)")


def main():
    # 出力フォルダを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # フォルダ内の WAV ファイル一覧を取得
    main_files = [f for f in os.listdir(MAIN_DIR) if f.endswith(".wav")]

    # 音声を処理
    for main_file in main_files:
        main_path = os.path.join(MAIN_DIR, main_file)
        process_audio_with_pitch_shift(main_path, OUTPUT_DIR, PITCH_SHIFT_VALUES)

if __name__ == "__main__":
    main()
