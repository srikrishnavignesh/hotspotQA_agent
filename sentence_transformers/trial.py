# %%
import chromadb
from sentence_transformers import SentenceTransformer
import commons
from operator import add
import commons
import json



client = chromadb.PersistentClient(path=commons.CHROMADB_PATH)

model = SentenceTransformer("BAAI/bge-base-en", device='cuda')

CONTEXT_LENGTH = model.tokenizer.model_max_length

class BGEEmbeddingFunction():
    def __init__(self, model):
        self.model = model

    #the BAAI model was trained in the following format 
    #query: What is the distance from earth too sun?
    #passage : some distance
    #by adding prefix of passage we get a better performance
    def __call__(self, texts: list[str]):
        queries = [f'query: {text}' for text in texts]
        return self.model.encode(
            queries,
            normalize_embeddings=True
        ).tolist()


collection = client.get_or_create_collection(name=commons.CHROMADB_COLLECTION_NAME)



model = BGEEmbeddingFunction(model)

#%%
queries = ['Townsend Coleman is an American voice actor?']
embeddings = model(queries)
res = collection.query(query_embeddings=embeddings,
                        include=["embeddings", "documents", "metadatas"],
                       n_results=10)


print(res)


# %%
