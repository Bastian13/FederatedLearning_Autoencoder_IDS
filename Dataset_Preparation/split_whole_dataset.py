import pandas as pd
import os
os.chdir("B:/B_Projekt2/CiC/BoTIoT") #Replace with your actual file path

# Configuration 
# run for each Dataset 2x Times: One Benign and one attack. 
INPUT_FILE = "CIC-BoT-IoT.csv"  # Replace with your actual file path
OUTPUT_FILE = 'all_attacks.csv' # Replace with your actual file path
CHUNK_SIZE = 100000  # Adjust based on your available RAM (100k rows at a time)

def filter_scanning_attacks(input_file, output_file, chunk_size):
    """
    Load large CSV in chunks, filter for scanning attacks, and save to new file. 
    
    Args:
        input_file:  Path to the input CSV file
        output_file: Path to the output CSV file
        chunk_size: Number of rows to process at a time
    """
    
    # Remove output file if it already exists
    if os.path. exists(output_file):
        os.remove(output_file)
    
    first_chunk = True
    total_rows_processed = 0
    total_scanning_attacks = 0
    
    print(f"Processing {input_file} in chunks of {chunk_size} rows...")
    
    # Process file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size,low_memory=False), 1):
        total_rows_processed += len(chunk)
        
        # Filter for scanning attacks
        # Adjust the column name and value based on your dataset structure
        # Common column names: 'attack', 'label', 'attack_type', 'category'
        
        # Option 1: If there's an 'attack' or 'category' column
        scanning_chunk = chunk[chunk['Label']== 1.0] # replace for different Datasets
        
        # Option 2: Alternative if column name is different (uncomment and modify)
        # scanning_chunk = chunk[chunk['label'].str.lower().str.contains('scan', na=False)] # replace for different Datasets
        
        # Option 3: If you need to filter for specific attack types
        # scanning_attacks_list = ['port_scan', 'os_scan', 'service_scan']
        # scanning_chunk = chunk[chunk['attack']. isin(scanning_attacks_list)] # replace for different Datasets
        
        total_scanning_attacks += len(scanning_chunk)
        
        # Save to output file
        if len(scanning_chunk) > 0:
            # Write header only for first chunk
            scanning_chunk.to_csv(
                output_file, 
                mode='a',  # Append mode
                header=first_chunk,
                index=False
            )
            first_chunk = False
        
        # Progress update
        print(f"Chunk {chunk_num}:  Processed {total_rows_processed: ,} rows, "
              f"Found {len(scanning_chunk):,} scanning attacks "
              f"(Total scanning attacks so far: {total_scanning_attacks:,})")
    
    print(f"\n✓ Processing complete!")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Total scanning attacks found: {total_scanning_attacks: ,}")
    print(f"Results saved to: {output_file}")
    
    # Display file sizes
    input_size = os.path.getsize(input_file) / (1024**3)  # GB
    if os.path.exists(output_file):
        output_size = os.path.getsize(output_file) / (1024**3)  # GB
        print(f"\nInput file size: {input_size:.2f} GB")
        print(f"Output file size: {output_size:.2f} GB")

if __name__ == "__main__":
    # Run the filtering
    filter_scanning_attacks(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)