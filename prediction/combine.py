import pandas as pd
import glob
import os

# 1. Set the path to the folder where you saved your NASA CSVs
input_path = './data/firms'  # Change this to your folder name
output_file = 'data/fires1.csv'

# 2. Get a list of all CSV files in that folder
all_files = glob.glob(os.path.join(input_path, "*.csv"))

# 3. Read and combine them
df_list = []
for filename in all_files:
    # We read each file and add a 'source_file' column just in case 
    # you need to know which sensor/year the data came from later.
    df = pd.read_csv(filename)
    df['origin_file'] = os.path.basename(filename)
    df_list.append(df)

# 4. Concatenate all dataframes into one
merged_df = pd.concat(df_list, axis=0, ignore_index=True)

# 5. Save to a single CSV
merged_df.to_csv(output_file, index=False)

print(f"Success! Combined {len(all_files)} files into {output_file}")