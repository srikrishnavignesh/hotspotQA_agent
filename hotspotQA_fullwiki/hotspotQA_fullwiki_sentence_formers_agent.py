import chromadb
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

client = chromadb.PersistentClient(path="./hotpot_chromadb")

model = SentenceTransformer("BAAI/bge-base-en", device='cuda')

CONTEXT_LENGTH = model.tokenizer.model_max_length

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
            batch_size=16,
            normalize_embeddings=True
        ).tolist()


collection = client.get_or_create_collection(
    name="hotspotQA_fullwiki_chunks",
    embedding_function=BGEEmbeddingFunction(model)
)



def ingest_jsonl_and_add_to_chromadb(docs_dir, batch_size=64):

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


DOCS_PATH = 'hotspotQA_fullwiki/pyserini/docs'
ingest_jsonl_and_add_to_chromadb(DOCS_PATH)

