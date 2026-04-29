import chromadb
from sentence_transformers import SentenceTransformer
import json
import commons
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from typing import Annotated
from operator import add
import commons
import json
from langchain_openrouter import ChatOpenRouter
import os
from datetime import timedelta
import time
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers  import CrossEncoder
from pydantic import BaseModel, ValidationError

class LLMResponse(BaseModel):
    answer : str = ''
    supporting_facts : dict[str, list[str]] = {}
    context_needed : list[str] = [] 


class BGEQueryEmbeddingFunction():
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


class HotSpotQA(TypedDict):
    question : str
    context : dict[str, dict[str, str]]
    prev_resp_comments:Annotated[list[str], add]
    
    actual_answer:str
    actual_supporting_facts: dict[str, list]
    
    raw_response:str
    retry_count:int
    
    pred_answer:str
    pred_supporting_facts: dict[str, list]

    provide_answer: bool
    question_id: str

    context_needed : list
    
    iteration : int
    response_proper : bool

    use_reranking :bool


def query_pyserini(titles, prev_context):

    for title in titles:
        hits = searcher.search(title, k=commons.PYSERINI_TOP_K_RETRIEVAL)
        for hit in hits:
            raw = json.loads(searcher.doc(hit.docid).raw()) # type: ignore
            title = raw['id']
            contents = raw['contents']
            
            sentences = contents.split('\n')
            if title not in prev_context:
                prev_context[title] = set()

            prev_context[title].update(set(sentences))
            
        
    return prev_context


def get_top_k_sentences(queries, title_sentences_map):
    updated_title_sentences_map = {}

    docs = []
    for title, sentences in title_sentences_map.items():
        for sentence in sentences:
            doc = f'title:{title}\ncontent:{sentence}'
            docs.append([None, doc])
    
    for query in queries:
        for doc in docs:
            doc[0] = query
          
        scores = cross_encoder.predict(docs)
        results = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:commons.TOP_K_RANK]
        
        for _, (_, doc) in results:
            contents = doc.split('\n')
            title = contents[0].removeprefix("title:")
            passage = contents[1].removeprefix("content:")
            if title not in updated_title_sentences_map:
                updated_title_sentences_map[title]  = set()
            
            updated_title_sentences_map[title].add(passage)
    
    return updated_title_sentences_map

def build_sentences_id(title_sentences_map):
    sentence_id = 0
    title_with_sentence_id_map = {}
    for title, sentences in title_sentences_map.items():
        if title not in title_with_sentence_id_map:
            title_with_sentence_id_map[title] = {}
        
        for sentence in sentences:
            sentence_id_str = f'{sentence_id}'
            title_with_sentence_id_map[title][sentence_id_str] = sentence
            sentence_id+=1
    
    return title_with_sentence_id_map


def build_context(prev_context, queries, use_reranking):

    title_sentences_map = query_pyserini(queries, prev_context)

    embeddings = model(queries)
    res = collection.query(query_embeddings=embeddings, n_results=commons.TOP_K_CONTEXTUAL_RETRIEVAL)

    if res['documents'] is None or res['metadatas'] is None:
        return build_sentences_id(title_sentences_map)
    
    for query_docs, query_metas in zip(res['documents'], res['metadatas']):
        for doc, query_meta in zip(query_docs, query_metas):
            
            title = query_meta['title']

            sentences = doc.split('\n')
            
            if title not in title_sentences_map:
                title_sentences_map[title] = set()
          
            title_sentences_map[title].update(set(sentences))
    
    if use_reranking:
        title_sentences_map = get_top_k_sentences(queries, title_sentences_map)

    return build_sentences_id(title_sentences_map)
    

def get_supporting_facts(supporting_facts, actual_context):
    titles = supporting_facts['title']
    sentence_ids = supporting_facts['sent_id']
    supporting_facts_map = {}
    for title, sentence_id in zip(titles, sentence_ids):
        pos = actual_context['title'].index(title)
        sentence = actual_context['sentences'][pos][sentence_id]
        
        if title not in supporting_facts_map:
            supporting_facts_map[title] = []
        
        supporting_facts_map[title].append(sentence)
        
    
    return supporting_facts_map


def query_llm(state:HotSpotQA):
    message = [
        {'role' : 'system', 'content' : commons.BASE_PROMPT},
        {'role' : 'user', 'content': commons.get_query_prompt_with_context(
            state['question'], state['context'], state['prev_resp_comments'], state['provide_answer']
            )
        }
    ]
    res = llm.invoke(message)
    return {'raw_response':res.content}

def is_answer_provided(answer):
    return not (answer.strip() == '' or  answer.lower() == 'null' or answer.lower() == 'none')


def validate_response(state:HotSpotQA):
    try:
        response = LLMResponse.model_validate_json(state['raw_response'])
    except ValidationError:
        return {
            'prev_resp_comments': [commons.INVALID_JSON_FORMAT_PROMPT],
            'retry_count': state['retry_count'] + 1,
            'response_proper' : False
            }
    
   
    for title, sentence_ids in response.supporting_facts.items():
        if title not in state['context']:
            return {
                'prev_resp_comments': [commons.get_invalid_titles_prompt(title)],
                'retry_count': state['retry_count'] + 1,
                'response_proper' : False
            }
        invalid_sentence_ids = set(sentence_ids) - set(state['context'][title].keys())
        if len(invalid_sentence_ids) > 0:
            return {
                'prev_resp_comments': [commons.get_invalid_sentence_ids_prompt(title, invalid_sentence_ids)],
                'retry_count': state['retry_count'] + 1,
                'response_proper' : False
            }

    if state['provide_answer']:
        if not is_answer_provided(response.answer):
            return {
                'prev_resp_comments': [commons.get_no_answer_in_respone_prompt()],
                'retry_count': state['retry_count'] + 1,
                'response_proper' : False
            }
        return {'pred_answer': response.answer,'pred_supporting_facts':response.supporting_facts, 'response_proper' : True}
    

    if is_answer_provided(response.answer) and not response.context_needed:
        return  {'pred_supporting_facts':response.supporting_facts, 'pred_answer': response.answer, 'response_proper' : True}


    return {'pred_supporting_facts':response.supporting_facts, 'context_needed': response.context_needed,'response_proper' : True}
    

def update_context(state:HotSpotQA):
    
    prev_supporting_facts = state['pred_supporting_facts']
    context = state['context']
    prev_context = {}
    for title, sentence_ids in prev_supporting_facts.items():
        sentences = context[title]
        pred_support_sentences = {sentences[sentence_id] for sentence_id in sentence_ids}
        
        if title not in prev_context:
            prev_context[title] = set()
        
        prev_context[title].update(pred_support_sentences)
       

    context_needed = state['context_needed']
    if context_needed:
        updated_context = build_context(prev_context, context_needed, state['use_reranking'])
    else:
        updated_context = build_sentences_id(prev_context)


    return {'context': updated_context}


def is_reponse_proper(state:HotSpotQA):
    if state['response_proper']:
        return 'response_proper'
    if state['retry_count'] == commons.MAX_RETRIES:
        return 'retry_maxed'
    return 'retry'

def write_retries_maxed_hotspotQA_to_file(state:HotSpotQA):
    file_path = commons.RETRIES_MAXED_FILE_PATH
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data.append(state)
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate() 
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([state], f, indent=4)

def move_state_fwd(state:HotSpotQA):    
    if state['iteration'] + 1 == commons.MAX_ITERATIONS:
        return {'iteration':state['iteration']+1, 'provide_answer': True}
    return {'iteration':state['iteration']+1}


def check_state(state:HotSpotQA):
    if state['iteration'] > commons.MAX_ITERATIONS or state['pred_answer'] != '':
        return 'end'
    else:
        return 'query'

def convert_sent_ids_to_sentence(state:HotSpotQA):
    pred_supporting_facts = state['pred_supporting_facts']
    converted_prediction_map = {}
    for title, pred_sentences_id in pred_supporting_facts.items():
        if title in state['context']:
            pred_sentences = []
            actual_sentences = state['context'][title]
            for pred_sentence_id in pred_sentences_id:
                if pred_sentence_id in actual_sentences:
                    pred_sentences.append(actual_sentences[pred_sentence_id])
            converted_prediction_map[title] = pred_sentences

    return {'pred_supporting_facts': converted_prediction_map}



def run_with_hybrid_agent(hotspotQA_test_data, use_reranking):

    graph = StateGraph(HotSpotQA)
    graph.add_node('query', query_llm)
    graph.add_node('validate_response', validate_response)
    graph.add_node('add_query_context', update_context)
    graph.add_node('convert_sent_ids_to_sentence', convert_sent_ids_to_sentence)


    graph.add_node('move_state_fwd', move_state_fwd)

    graph.add_node('write_retries_maxed_hotspotQA_to_file', write_retries_maxed_hotspotQA_to_file)
    
    graph.set_entry_point('query')

    graph.add_edge('query', 'validate_response')



    graph.add_conditional_edges('validate_response', is_reponse_proper, {
        'retry':'query','retry_maxed':'write_retries_maxed_hotspotQA_to_file',
        'response_proper':'move_state_fwd'
        }
    )


    graph.add_conditional_edges('move_state_fwd', check_state, {
        'query':'add_query_context','end':'convert_sent_ids_to_sentence'
        }
    )

    graph.add_edge('add_query_context', 'query')

    graph.add_edge('write_retries_maxed_hotspotQA_to_file', 'convert_sent_ids_to_sentence')

    graph.add_edge('convert_sent_ids_to_sentence', END)

    
    app = graph.compile()

    answer_matched = 0
    supporting_facts_matched = 0
    time_sum = timedelta(seconds=0)
    retry_sum = 0
    
    n = len(hotspotQA_test_data)
   
    print('running hybrid based QA')
    observations = []
    for indx, test_data in enumerate(hotspotQA_test_data):
        start = time.perf_counter()
        state : HotSpotQA = {
            'question' : test_data['question'],
            'context': build_context({}, [test_data['question']], use_reranking),
            'actual_answer' : test_data['answer'],
            'actual_supporting_facts': get_supporting_facts(test_data['supporting_facts'], test_data['context']),
            'prev_resp_comments' : [],
            'raw_response':'',
            'retry_count': 0,
            'pred_answer': '',
            'pred_supporting_facts' : {},
            'provide_answer': False,
            'question_id': test_data['id'],
            'context_needed':[],
            'iteration':1,
            'response_proper' : False,
            'use_reranking':use_reranking
        }

        res = app.invoke(state)
        end = time.perf_counter()
        matched = False
        if res['pred_answer'].lower() == res['actual_answer'].lower():
            answer_matched+=1
            matched = True
        
        if res['pred_supporting_facts'].keys() == res['actual_supporting_facts'].keys():
            supporting_facts_matched+=1
        
        observation =  {'index' : indx, 'question_id': res['question_id'], 'actual_answer' : res['actual_answer'], 
                        'pred_answer' : res['pred_answer'], 'actual_supporting_facts': res['actual_supporting_facts'], 
                        'pred_supporting_facts': res['pred_supporting_facts'], 'matched':matched
                        }
        observations.append(observation)
        
        print(observation)

        delta = timedelta(seconds=(end - start))
        time_sum += delta

        retry_sum+=state['retry_count']
        

    metrics = {'metrics': 
                {
                    'n' : n, 'exact_matched_answers' : answer_matched,
                    'exact_matched_doc_keys' : supporting_facts_matched,
                    'avg_time_in_seconds' : time_sum.total_seconds() / n, 'retry_count' : retry_sum / n,
                    'total_time_taken_in_seconds' : time_sum.total_seconds(),'total_retry_count' : retry_sum
                }
              }
    
    observations.append(metrics)

    print(metrics)

    if use_reranking:
       filename = commons.HYBRID_RESULTS_WITH_RERANKING
    else:
      filename = commons.HYBRID_RESULTS_WITHOUT_RERANKING

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(observations, f, ensure_ascii=False, indent=2)




if __name__ == '__main__':
    api_key = os.getenv('OPEN-ROUTER-API-KEY')

    llm = ChatOpenRouter(
        model="qwen/qwen3-32b",
        api_key=api_key, # type: ignore
        temperature=0
    )

    client = chromadb.PersistentClient(path=commons.CHROMADB_PATH)

    model = BGEQueryEmbeddingFunction(SentenceTransformer(commons.BAA_BASE, device='cuda', local_files_only=True))
    collection = client.get_or_create_collection(name=commons.CHROMADB_COLLECTION_NAME)

    
    with open(commons.TEST_DATA_PATH, 'r', encoding='utf-8') as fp:
        hotspotQA_test_data = json.load(fp) 

    
    searcher = LuceneSearcher(commons.PYSERINI_INDEX_FILE_DOX)

    cross_encoder = CrossEncoder(commons.BAA_BASE_RERANKER, device='cuda', local_files_only=True)


    run_with_hybrid_agent(hotspotQA_test_data, True)

    run_with_hybrid_agent(hotspotQA_test_data, False)
