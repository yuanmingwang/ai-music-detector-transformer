import pandas as pd
import os


def process_data(data_dir, folder_name, csv_name, target, limit=None):
    # 1. 读取并清理
    df = pd.read_csv(f"{data_dir}/{csv_name}")
    df["filename"] = df["filename"].astype(str).str.strip()
    
    # 2. 物理文件检查 (核心功能)
    audio_path = f"{data_dir}/{folder_name}"
    existing_files = {os.path.splitext(f)[0] for f in os.listdir(audio_path) if f.endswith('.mp3')}
    df = df[df["filename"].isin(existing_files)]
    
    # 3. 基础过滤 (duration & vocal)
    df = df[(df.duration >= 30) & (df.no_vocal == False)]
    
    # 4. 数量限制
    if limit:
        df = df.head(limit)
        
    # 5. 设置路径和标签
    df["filepath"] = f"{audio_path}/" + df.filename + ".mp3"
    df["target"] = target
    return df

if __name__ == "__main__":
    DATA_DIR = "./dataset"
    LIMIT = 100  # 如果不需要限制，设为 None

    # 处理两类数据
    real_df = process_data(DATA_DIR, "real_songs", "real_songs.csv", 0, limit=LIMIT)
    fake_df = process_data(DATA_DIR, "fake_songs", "fake_songs.csv", 1, limit=LIMIT)

    # 合并
    df = pd.concat([real_df, fake_df])

    # 导出 (保持原样)
    for split in ['train', 'valid', 'test']:
        df[df.split == split].to_csv(f"{split}.csv", index=False)