
from pathlib import Path
import polars as pl 
import os




def write_train_data(train_data_path, distractor_input_json_path, fullwiki_input_json_path):
    df = pl.read_ipc_stream(train_data_path)

    filtered_df = df.filter(pl.col("level") == "hard").sample(n=200, seed=42)    

    if not os.path.exists(distractor_input_json_path):
        filtered_df.to_pandas().to_csv(distractor_input_json_path, index=False)
    
    if not os.path.exists(fullwiki_input_json_path):
        filtered_df.to_pandas().to_csv(fullwiki_input_json_path, index=False)
        
            
DISTRACTOR_INPUT_JSON_PATH = 'hotspotQA_distractor/train.json'
FULLWIKI_INPUT_JSON_PATH = 'hotspotQA_fullwiki/train.json'

TRAIN_DATA_PATH = 'data-00000-of-00002.arrow'

write_train_data(TRAIN_DATA_PATH, DISTRACTOR_INPUT_JSON_PATH, FULLWIKI_INPUT_JSON_PATH)



