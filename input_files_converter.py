
from pathlib import Path
import polars as pl 
import json
import re
from pathlib import Path
import json
from pathlib import Path


def parse_train_data(input_data_path):
    
    with open(input_data_path, encoding='utf-8') as f:
        train_data = json.load(f)

    title_to_sentences_map = {}
    for data in train_data:
        context = data['context']
        for indx, title in enumerate(context['title']):
            if title not in title_to_sentences_map:
                title_to_sentences_map[title] = set()
            
            title_to_sentences_map[title].update(context['sentences'][indx])
    
    return title_to_sentences_map



def clean(sentence):
    sentence =  re.sub("<a href.*?>", '', sentence)
    sentence = sentence.replace("</a>", '')
    sentence = sentence.replace("&amp", "&")
    sentence = re.sub(r"\s+", ' ', sentence)
    return sentence.strip()

class RotatingJSONLWriter:
    def __init__(self, base_filename,files_prefix, max_mb=50):
        self.max_bytes = max_mb * 1024 * 1024
        self.file_count = 0

        self.file_name = f'{base_filename}/{files_prefix}'

        file_path = Path(self.file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_file  = None


    def _start_new_file(self):
        self.file_count += 1
        filename = f"{self.file_name}{self.file_count}.jsonl"
        self.current_file = open(filename, 'w', encoding='utf-8')
        print(f"Started new shard: {filename}")
     

    def write(self, data):
        
        if self.current_file is None:
            raise ValueError("start with context before writing")

        line = json.dumps(data, ensure_ascii=False) + "\n"

        self.current_file.write(line)
        
        if self.current_file.tell() > self.max_bytes:
             self.current_file.close()
             self._start_new_file()


    def __enter__(self):
        self._start_new_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_file:
            self.current_file.close()


def convert_to_json_l_file(input_json_path, dest_path, dest_files_prefix):
    title_to_sentences_map = parse_train_data(input_json_path)

    with RotatingJSONLWriter(dest_path, dest_files_prefix) as jsonl_file:
        for title, sentences in title_to_sentences_map.items():
            
            cleaned_sentences = []
            for sentence in sentences:

                cleaned_sentence = clean(sentence)
                if cleaned_sentence == '':
                    continue
                cleaned_sentences.append(cleaned_sentence)

            #new line separates the sentences
            para = '\n'.join(cleaned_sentences)
        
            output_obj = {'id' : f'{title}', 'contents': para}

            jsonl_file.write(output_obj)



def convert_to_json(train_data_path, input_json_path, train_json_path, test_json_path):
    df = pl.read_ipc_stream(train_data_path)

    test_df = df.filter(pl.col("level") == "hard").sample(n=200, seed=42)    

    train_df = df.filter(~pl.col("id").is_in(test_df.get_column("id")))

    input_df = df

    
    with open(input_json_path, 'w', encoding='utf-8') as f:
        json.dump(input_df.to_dicts(), f, ensure_ascii=False, indent=2)

    
    # with open(train_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(train_df.to_dicts(), f, ensure_ascii=False, indent=2)
    
    
    # with open(test_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(test_df.to_dicts(), f, ensure_ascii=False, indent=2)
    

        
            

INPUT_ARROW_PATH = 'data-00000-of-00002.arrow'

INPUT_JSON_PATH = 'input.json'

TEST_JSON_PATH = 'test.json' 

TRAIN_JSON_PATH = 'train.json'

JSON_L_DOCS_PATH = 'json_l_docs'

JSON_L_FILES_PREFIX = 'doc_'

convert_to_json(INPUT_ARROW_PATH, INPUT_JSON_PATH,  TRAIN_JSON_PATH, TEST_JSON_PATH)

# convert_to_json_l_file(INPUT_JSON_PATH, JSON_L_DOCS_PATH, JSON_L_FILES_PREFIX)



