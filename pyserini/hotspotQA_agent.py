from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from typing import Annotated
from operator import add
import commons
from pyserini.search.lucene import LuceneSearcher
import json
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr
import polars as pl 
import os
from datetime import timedelta
import time

with open(commons.TEST_DATA_PATH, 'r', encoding='utf-8') as fp:
    hotspotQA_test_data = json.load(fp) 

searcher = LuceneSearcher(commons.PYSERINI_INDEX_FILE_DOX)

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


def build_context(titles):

    title_sentences_map = {}
    
    for title in titles:
        hits = searcher.search(title, k=commons.TOP_K_RETRIEVAL)
        for hit in hits:
            raw = json.loads(searcher.doc(hit.docid).raw()) # type: ignore
            title = raw['id']
            contents = raw['contents']
            
            sentences = contents.split('\n')
            
            sentence_id_map = {}
            for indx,sentence in enumerate(sentences):
                sentence_id = f'{indx}'
                sentence_id_map[sentence_id] = sentence
            
            title_sentences_map[title] = sentence_id_map
    
    return title_sentences_map

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


def validate_response(state:HotSpotQA):
   
    try:
        json_response = json.loads(state['raw_response'])
    except json.JSONDecodeError:
        return {
            'prev_resp_comments': [commons.INVALID_JSON_FORMAT_PROMPT],
            'retry_count': state['retry_count'] + 1
            }
    
    updated_supporting_facts = {}
    if 'supporting_facts' in json_response:
        supporting_facts = json_response['supporting_facts']
        
        for title, sentence_ids in supporting_facts.items():
            str_sentence_ids = [f'{id}' if not isinstance(id, str) else id for id in sentence_ids]
            updated_supporting_facts[title] = str_sentence_ids


        for title, sentence_ids in updated_supporting_facts.items():
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
        if 'answer'not in json_response or json_response['answer'].strip() == '' or  json_response['answer'] is None or json_response['answer'] == 'null' or json_response['answer'].lower() == 'none':
            return {
                'prev_resp_comments': [commons.get_no_answer_in_respone_prompt()],
                'retry_count': state['retry_count'] + 1,
                'response_proper' : False
            }
        return {'pred_answer': json_response['answer'],'pred_supporting_facts':updated_supporting_facts, 'response_proper' : True}
    
    if 'context_needed' in json_response and isinstance(json_response['context_needed'], list):
        return {'pred_supporting_facts':updated_supporting_facts, 'context_needed': json_response['context_needed'], 'response_proper' : True}
    
    return {'pred_supporting_facts':updated_supporting_facts, 'response_proper' : True}

def update_context(state:HotSpotQA):

    context_needed = state['context_needed']
    new_context = {}
    if context_needed:
        new_context = build_context(context_needed)

    prev_supporting_facts = state['pred_supporting_facts']
    context = state['context']
    
    for title, sentence_ids in prev_supporting_facts.items():
        sentences = context[title]
        new_sentences = {sentence_id:sentences[sentence_id] for sentence_id in sentence_ids}
        if title in new_context:
            new_context[title].update(new_sentences)
        else:
            new_context[title] = new_sentences

    return {'context': new_context}


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
    if state['iteration'] > commons.MAX_ITERATIONS:
        return 'end'
    else:
        return 'query'

def convert_sent_ids_to_sentence(state:HotSpotQA):
    pred_supporting_facts = state['pred_supporting_facts']
    converted_prediction_map = {}
    for title, sentences_id in pred_supporting_facts.items():
        pred_sentences = []
        sentences = state['context'][title]
        for sentence_id in sentences_id:
            sentence = sentences[sentence_id]
            pred_sentences.append(sentence)
        converted_prediction_map[title] = pred_sentences

    return {'pred_supporting_facts': converted_prediction_map}




def run_with_pyserini():

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
        'response_proper':'add_query_context'
        }
    )

    graph.add_edge('add_query_context', 'move_state_fwd')

    graph.add_conditional_edges('move_state_fwd', check_state, {
        'query':'query','end':'convert_sent_ids_to_sentence'
        }
    )

    graph.add_edge('write_retries_maxed_hotspotQA_to_file', END)

    graph.add_edge('convert_sent_ids_to_sentence', END)

    
    app = graph.compile()

    answer_matched = 0
    supporting_facts_matched = 0
    time_sum = timedelta(seconds=0)
    retry_sum = 0
    
    n = len(hotspotQA_test_data)
   
    print('running pyserini based QA')
    observations = []
    for indx, test_data in enumerate(hotspotQA_test_data):
        start = time.perf_counter()
        state : HotSpotQA = {
            'question' : test_data['question'],
            'context': {},
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
            'response_proper' : False
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

    with open(commons.PYSERSINI_BASED_QUERY_RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(observations, f, ensure_ascii=False, indent=2)


api_key = os.getenv('OPEN-ROUTER-API-KEY')

llm = ChatOpenRouter(
  model="qwen/qwen3-32b",
  api_key=api_key, # type: ignore
  temperature=0
)

run_with_pyserini()