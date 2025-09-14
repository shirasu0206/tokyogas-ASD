import os
import subprocess

def extract_audio_from_mp4(input_dir, output_dir):
    """
    指定したディレクトリ内のすべてのMP4ファイルからMP3音声を抽出して保存する。
    
    :param input_dir: MP4ファイルが格納されているディレクトリのパス
    :param output_dir: MP3ファイルを保存するディレクトリのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 指定ディレクトリ内のファイルを取得
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp4"):
            input_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".mp3"
            output_path = os.path.join(output_dir, output_file_name)
            
            # ffmpeg コマンドを実行
            command = [
                "ffmpeg", "-i", input_path, "-q:a", "0", "-map", "a", output_path, "-y"
            ]
            
            print(f"Extracting MP3 from: {input_path} -> {output_path}")
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("処理が完了しました！")

# 使用例
if __name__ == "__main__":
    input_directory = "./input"  # MP4ファイルがあるディレクトリ
    output_directory = "./output"  # MP3を保存するディレクトリ
    
    extract_audio_from_mp4(input_directory, output_directory)
