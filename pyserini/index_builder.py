import json
import re
from pathlib import Path
import json
from pathlib import Path
import commons


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
    def __init__(self, base_filename,files_prefix, max_mb=500):
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


def create_json_l_file():
    title_to_sentences_map = parse_train_data(commons.INPUT_DATA)
    dest_path = commons.PYSERINI_DOCS
    dest_files_prefix = commons.PYSERINI_FILES_PREFIX

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

create_json_l_file()