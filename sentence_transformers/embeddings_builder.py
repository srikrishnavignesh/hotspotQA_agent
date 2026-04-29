import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np
from tqdm import tqdm
import torch
import commons


def ingest_data_and_embeddings_to_chromadb(data_and_embed_doc_paths, collection, batch_size=64):

    completed_files = 0
    ids = []
    texts = []
    metadatas = []
    embeddings = []
    for data_path, embed_doc_path in data_and_embed_doc_paths:

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        embedded_doc = np.load(embed_doc_path)


        for indx,content in enumerate(data):

            id = content["id"] 
            chunk = content["chunk"]
            title = content["title"]
            
            embedding = embedded_doc[indx]

        
            ids.append(id)
            texts.append(chunk)

            metadatas.append({
                "title": title
            })
            
          
            embeddings.append(embedding)

            if len(texts) == batch_size:
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

                ids, texts, metadatas, embeddings = [], [], [], []

        completed_files+=1
        print(f'completed files : {completed_files}')

    if texts:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

def get_no_of_words(s):
    return len(s.split())

def get_word_freq(sentences):
    words_num = []
    for indx, sentence in enumerate(sentences):
        total_words = get_no_of_words(sentence)
        words_num.append(total_words)

    return  words_num




class BGEEmbeddingFunction():
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    #the BAAI model was trained in the following format 
    #query: What is the distance from earth too sun?
    #passage : some distance
    #by adding prefix of passage we get a better performance
    def __call__(self, texts: list[str]):
        with torch.no_grad():
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True
            )



def create_data_and_embeddings_from_jsonl(embedder, file_id_to_path_map, dest_dir, data_file_prefix, 
                                          embed_doc_file_prefix, batch_size=3500, overlap=1):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    MAX_WORDS = 400 
 
    for indx, file_path in file_id_to_path_map.items():
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        chunks_meta = []
        embeddings = []
        chunks_for_embed = []
        
        for line in tqdm(lines, desc='processing_lines'):
            data = json.loads(line)

            title = data["id"] 
            text = data["contents"]
            sentences = text.split('\n')

            words_num = get_word_freq(sentences)

            s = 0
            added_till = -1
            while s < len(sentences):
                
                e = s+1
                sum_words = words_num[s]
                while e < len(words_num) and (words_num[e] + sum_words) <= MAX_WORDS:
                    sum_words += words_num[e]
                    e+=1
                
                if e > added_till:
                    
                    chunk_for_retrieval = '\n'.join(sentences[s:e])
                
                    chunk_for_embed = f"title: {title}\n passage:{''.join(sentences[s:e])}"

                    chunk_id = f"{title}_{s}"

                    chunks_meta.append({'id' : chunk_id, 'title' : title, 'chunk':chunk_for_retrieval})
                    
                    chunks_for_embed.append(chunk_for_embed)

                    if len(chunks_for_embed) == batch_size:
                        embeddings.extend(embedder(chunks_for_embed))
                        chunks_for_embed = []
                
                if e == len(sentences):
                    s = e
                elif  e - s > overlap:
                    s = e - overlap
                    added_till = e-1
                else:
                    s+=1
                    added_till = e-1

        if len(chunks_for_embed):
            embeddings.extend(embedder(chunks_for_embed))


        with open(f'{dest_dir}/{data_file_prefix}{indx}.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_meta, f, ensure_ascii=False)
        
        np.save(f'{dest_dir}/{embed_doc_file_prefix}{indx}.npy', embeddings)


#input must be jsonl files
def insert_vectors_to_chromadb(embedder):


    dest_dir='sentence_transformers/data_and_embeddings'

    docs_src_dir = 'json_l_docs'
    docs_file_prefix = 'doc_'

    docs_count = len([f for f in os.listdir(docs_src_dir) if os.path.isfile(os.path.join(docs_src_dir, f))])
    
    data_file_prefix='data_'
    embed_doc_file_prefix = 'embedded_doc_'


    file_id_to_path_map = {counter:f'{docs_src_dir}/{docs_file_prefix}{counter}.jsonl' for counter in range(1, docs_count+1)}  
    
    # create_data_and_embeddings_from_jsonl(embedder, file_id_to_path_map, dest_dir, data_file_prefix, embed_doc_file_prefix)
    
    client = chromadb.PersistentClient(path=commons.CHROMADB_PATH)
    collection = client.get_or_create_collection(name=commons.CHROMADB_COLLECTION_NAME)


    data_and_embed_doc_paths = []
    for id in range(1, 3):
        data_file_path = f'{dest_dir}/{data_file_prefix}{id}.json'
        embed_doc_file_path = f'{dest_dir}/{embed_doc_file_prefix}{id}.npy'
        data_and_embed_doc_paths.append((data_file_path, embed_doc_file_path))

    ingest_data_and_embeddings_to_chromadb(data_and_embed_doc_paths, collection, batch_size=5000)


if __name__ == '__main__':

    model = SentenceTransformer(commons.BAA_BASE, device='cuda')

    model.max_seq_length = 512

    model.half()

    batch_size = 3500
    embedder = BGEEmbeddingFunction(model, batch_size)


    insert_vectors_to_chromadb(embedder)



    
