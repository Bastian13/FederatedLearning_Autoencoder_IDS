import pandas as pd
import os

# Configuration
#INPUT_FILE = "CIC-BoT-IoT.csv"  # Replace with your actual file path
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
# Load the large CSV
#auskommentiert damit ich nicht ausversehen nochmal ausführe
os.chdir("B:/B_Projekt2/CiC/IoTID20")

input_file = "Benign.csv"

output_dir = "small_IoTID20_dataset_benign_clean_noleak"


def shuffle_and_split_csv(input_file, output_dir, chunk_size=1000, random_seed=42):
    """
    Shuffle the entire dataset first, then split into non-overlapping chunks.

    Parameters:
        input_file (str): Path to CSV file
        output_dir (str): Folder to save splits
        chunk_size (int): Number of rows per file
        random_seed (int): Seed for reproducibility (optional)
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Load full dataset
    df = pd.read_csv(input_file)
    df= df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna()
    df=df.drop_duplicates()
    df=df.fillna(0)  
    # 2️⃣ Shuffle ENTIRE dataset
    # 3️⃣ Split sequentially (no overlap possible)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df, glo = train_test_split(df,test_size=0.2)
    df = df.sort_values(by='Flow_Duration')

    total_rows = len(df)
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk_number = i // chunk_size + 1
        chunk.to_csv(f"{output_dir}/split_{chunk_number}.csv", index=False)

    print(f"Shuffled and created {chunk_number} files.")
    total_rows = len(glo)
    for i in range(0, total_rows, chunk_size):
        chunk = glo.iloc[i:i + chunk_size]
        chunk_number = i // chunk_size + 1
        chunk.to_csv(f"{output_dir}/glo_split_{chunk_number}.csv", index=False)
    print(f"Shuffled and created {chunk_number} files.")


shuffle_and_split_csv(input_file,output_dir)