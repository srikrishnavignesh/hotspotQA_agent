import tarfile
import bz2
import json
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path


def extract_json_objects_from_tar(tar_path):
    with tarfile.open(tar_path, 'r:bz2') as tar:
        for member in tar:
                if member.isfile() and member.name.endswith('.bz2'):
                    inner_bz2 = tar.extractfile(member)
                    if inner_bz2:
                        with bz2.open(inner_bz2) as f:
                            for line in f:
                                yield json.loads(line.decode('utf-8'))
                        print(f"Finished processing {member.name}")


def clean(sentence):
    sentence =  re.sub("<a href.*?>", '', sentence)
    sentence = sentence.replace("</a>", '')
    sentence = sentence.replace("&amp", "&")
    sentence = re.sub(r"\s+", ' ', sentence)
    return sentence.strip()

class RotatingJSONLWriter:
    def __init__(self, base_filename, max_mb=500):
        self.max_bytes = max_mb * 1024 * 1024
        self.file_count = 0

        self.file_name = f'{base_filename}/doc'

        file_path = Path(self.file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_file  = None


    def _start_new_file(self):
        self.file_count += 1
        filename = f"{self.file_name}_{self.file_count}.jsonl"
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


def create_json_l_file(tar_path, destination_path):
    
    with RotatingJSONLWriter(destination_path) as jsonl_file:
        for obj in extract_json_objects_from_tar(tar_path):
        
            id = obj['id']
            title = obj['title']
            sentences = []
            for sentence in obj['text']:
                sentence = ''.join(sentence)
                sentence = clean(sentence)
                sentence = sentence+'(sentence-ends)'
                sentences.append(sentence)

            para = ''.join(sentences)
        
            output_obj = {'id' : f'{id}_{title}', 'contents': para}

            jsonl_file.write(output_obj)




def create_pyserini_index():
    FULL_WIKI_TAR_PATH = 'enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2'

    PYSERINI_DOCS = 'hotspotQA_fullwiki/pyserini/docs'
    create_json_l_file(FULL_WIKI_TAR_PATH, PYSERINI_DOCS)


class BGEEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    #the BAAI model was trained in the following format 
    #query: What is the distance from earth too sun?
    #passage : some distance
    #by adding prefix of passage we get a better performance
    def __call__(self, texts: list[str]):
        texts = ['passage:'+text for text in texts]
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True
        ).tolist()

def ingest_jsonl_and_add_to_chromadb(docs_dir, collection, batch_size=64):

    docs_dir_path = Path(docs_dir)
    completed_files = 0
    for file_path in docs_dir_path.iterdir():

        ids = []
        documents = []
        metadatas = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                id = data["id"] 
                text = data["contents"]
                chunks = text.split('(sentence-ends)')

                title = chunks[0]

                for indx, chunk in enumerate(chunks[1:]):
                
                    chunk_text_with_title = f"{title}\n{chunk}"

                    sentence_id = indx

                    chunk_id = f"{id}_{sentence_id}"

                    ids.append(chunk_id)
                    documents.append(chunk_text_with_title)
                    metadatas.append({
                        "title": title,
                        "sentence_id": sentence_id,
                    })


                
                    if len(documents) >= batch_size:
                        collection.add(
                            ids=ids,
                            documents=documents,
                            metadatas=metadatas
                        )

                        ids, documents, metadatas = [], [], []


        if documents:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        completed_files+=1
        print(f'completed files : {completed_files}')

def create_vectors_in_chromadb():
    client = chromadb.PersistentClient(path="./hotspot_chromadb")

    model = SentenceTransformer("BAAI/bge-base-en", device='cuda')

    collection = client.get_or_create_collection(
        name="hotspotQA_fullwiki_chunks",
        embedding_function=BGEEmbeddingFunction(model)
    )

    DOCS_PATH = 'hotspotQA_fullwiki/pyserini/docs'
    ingest_jsonl_and_add_to_chromadb(DOCS_PATH, collection, batch_size=2048)

create_vectors_in_chromadb()
