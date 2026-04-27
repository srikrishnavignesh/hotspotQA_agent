import polars as pl
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr
import commons
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from typing import Annotated
from operator import add
import os
import time
from datetime import timedelta

with open(commons.TEST_DATA_PATH, 'r', encoding='utf-8') as fp:
    hotspotQA_train_data = json.load(fp) 

                          
class HotSpotQA(TypedDict):
    question : str
    context : dict[str, dict[str, str]]
    prev_resp_comments:Annotated[list[str], add]
    
    actual_answer:str
    actual_supporting_facts: dict[str, list]
    
    resp_answer:str
    resp_supporting_facts: dict[str, list]
    raw_response:str
    retry_count:int
    
    pred_answer:str
    pred_supporting_facts: dict[str, list]

    provide_answer: bool

    question_id: str


def get_context(context):
    titles = context['title']
    sentences = context['sentences']
    title_sentences_map = {}
    for indx, title in enumerate(titles):
        sentence_id_map = {}
        for indx,sentence in enumerate(sentences[indx]):
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
        {'role' : 'user', 'content': commons.get_query_prompt_with_context(state['question'], state['context'], 
                                                state['prev_resp_comments'], state['provide_answer'])}
    ]
    res = llm.invoke(message)
    return {'raw_response':res.content}

def query_llm_with_no_context(state:HotSpotQA):
    message = [
        {'role' : 'system', 'content' : commons.NO_CONTEXT_PROMPT},
        {'role' : 'user', 'content': commons.get_query_prompt_without_context(
            state['question'], state['prev_resp_comments'])}
    ]
    res = llm.invoke(message)
    return {'raw_response':res.content}


def validate_supporting_facts(state:HotSpotQA):
    try:
        json_response = json.loads(state['raw_response'])
    except json.JSONDecodeError:
        return {
            'prev_resp_comments': [commons.INVALID_JSON_FORMAT_PROMPT],
            'retry_count': state['retry_count'] + 1
            }
    
    supporting_facts = json_response['supporting_facts']
    
    updated_supporting_facts = {}
    for title, sentence_ids in supporting_facts.items():
        str_sentence_ids = [f'{id}' if not isinstance(id, str) else id for id in sentence_ids]
        updated_supporting_facts[title] = str_sentence_ids


    for title, sentence_ids in updated_supporting_facts.items():
        if title not in state['context']:
            return {
                'prev_resp_comments': [commons.get_invalid_titles_prompt(title)],
                'retry_count': state['retry_count'] + 1
            }
        invalid_sentence_ids = set(sentence_ids) - set(state['context'][title].keys())
        if len(invalid_sentence_ids) > 0:
            return {
                'prev_resp_comments': [commons.get_invalid_sentence_ids_prompt(title, invalid_sentence_ids)],
                'retry_count': state['retry_count'] + 1
            }
        
    return {'pred_supporting_facts':updated_supporting_facts}

def update_context_with_prev_supporting_facts(state:HotSpotQA):
    prev_supporting_facts = state['pred_supporting_facts']
    context = state['context']
    new_context = {}
    for title, sentence_ids in prev_supporting_facts.items():
        sentences = context[title]
        new_sentences = {sentence_id:sentences[sentence_id] for sentence_id in sentence_ids}  
        new_context[title] = new_sentences
    
    return {'context': new_context, 'provide_answer': True}

def validate_answer(state:HotSpotQA):
    try:
        json_response = json.loads(state['raw_response'])
    except json.JSONDecodeError:
        return {
            'prev_resp_comments': [commons.INVALID_JSON_FORMAT_PROMPT],
            'retry_count': state['retry_count'] + 1
            }
    
    pred_answer = json_response['answer']
    if pred_answer is None or pred_answer.strip() == ''  or pred_answer.lower() == 'none' or pred_answer.lower() == 'null':
        return {
            'prev_resp_comments': [commons.get_no_answer_in_respone_prompt()],
            'retry_count': state['retry_count'] + 1
        }
    return {'pred_answer': pred_answer, 'prev_resp_comments':[]}

def check_answer(state:HotSpotQA):
        if state['pred_answer'] != '':
            return 'end'
        if state['retry_count'] == commons.MAX_RETRIES:
            return 'retry_maxed'
        return 'retry'
    

def check_for_supporting_facts(state:HotSpotQA):
    if len(state['pred_supporting_facts']) > 0:
        return 'end'
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



def run_single_hop():

    graph = StateGraph(HotSpotQA)
    graph.add_node('get_answer', query_llm)
    graph.add_node('validate_supporting_facts', validate_supporting_facts)
    graph.add_node('write_retries_maxed_hotspotQA_to_file', write_retries_maxed_hotspotQA_to_file)
    graph.add_node('validate_answer', validate_answer)

    graph.add_node('convert_ids_to_sentences', convert_sent_ids_to_sentence)

    graph.set_entry_point('get_answer')
    graph.add_edge('get_answer', 'validate_supporting_facts')
    graph.add_edge('write_retries_maxed_hotspotQA_to_file', END)

  
   


    graph.add_conditional_edges('validate_supporting_facts',check_for_supporting_facts, {
        'retry':'get_answer','retry_maxed':'write_retries_maxed_hotspotQA_to_file','end': 'validate_answer'}
        )
    
    graph.add_conditional_edges('validate_answer', check_answer, {
        'retry': 'get_answer',
        'retry_maxed': 'write_retries_maxed_hotspotQA_to_file',
        'end': 'convert_ids_to_sentences'
    })

    graph.add_edge('convert_ids_to_sentences', END)

    app = graph.compile()

    answers_matched = 0
    supporting_facts_keys_matched = 0
    time_sum = timedelta(seconds=0)
    retry_sum = 0
    n = len(hotspotQA_train_data)
    observations = []

    print('running single hop')
    for indx, train_data in enumerate(hotspotQA_train_data):
        
        start = time.perf_counter()
        state : HotSpotQA = {
            'question' : train_data['question'],
            'context': get_context(train_data['context']),
            'actual_answer' : train_data['answer'],
            'actual_supporting_facts': get_supporting_facts(train_data['supporting_facts'], train_data['context']),
            'prev_resp_comments' : [],
            'resp_answer': '',
            'resp_supporting_facts': {},
            'raw_response':'',
            'retry_count': 0,
            'pred_answer': '',
            'pred_supporting_facts' : {},
            'provide_answer': True,
            'question_id': train_data['id']
        }

        res = app.invoke(state)
        end = time.perf_counter()

        matched = False
        if res['pred_answer'].lower() == res['actual_answer'].lower():
            answers_matched+=1
            matched = True
        
        if res['pred_supporting_facts'].keys() == res['actual_supporting_facts'].keys():
            supporting_facts_keys_matched+=1
    
           
        observation =  {'index':indx, 'question_id': res['question_id'], 'actual_answer' : res['actual_answer'], 
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
                'n' : n, 'exact_matched_answers' : answers_matched,
                'exact_matched_doc_keys' : supporting_facts_keys_matched,
                'avg_time_in_seconds' : time_sum.total_seconds() / n, 'retry_count' : retry_sum / n,
                'total_time_taken_in_seconds' : time_sum.total_seconds(),'total_retry_count' : retry_sum
            }
            }

    
    print(metrics)

    observations.append(metrics)
    with open(commons.SINGLE_HOP_RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(observations, f, ensure_ascii=False, indent=2)
            

def run_multi_hop():

    graph = StateGraph(HotSpotQA)
    graph.add_node('get_context_for_reasoning', query_llm)
    graph.add_node('validate_supporting_facts', validate_supporting_facts)
    graph.add_node('write_retries_maxed_hotspotQA_to_file', write_retries_maxed_hotspotQA_to_file)
    graph.add_node('update_context_with_prev_supporting_facts', update_context_with_prev_supporting_facts)
    graph.add_node('validate_answer', validate_answer)
    
    graph.add_node('get_answer', query_llm)

    graph.add_node('convert_ids_to_sentences', convert_sent_ids_to_sentence)

    graph.set_entry_point('get_context_for_reasoning')
    graph.add_edge('get_context_for_reasoning', 'validate_supporting_facts')
    graph.add_edge('update_context_with_prev_supporting_facts', 'get_answer')
    graph.add_edge('get_answer', 'validate_answer')

    graph.add_edge('write_retries_maxed_hotspotQA_to_file', END)


    graph.add_conditional_edges('validate_supporting_facts', check_for_supporting_facts, {
        'retry':'get_context_for_reasoning','retry_maxed':'write_retries_maxed_hotspotQA_to_file',
        'end':'update_context_with_prev_supporting_facts'
        }
    )

    
    graph.add_conditional_edges('validate_answer', check_answer, {
        'retry': 'get_answer',
        'retry_maxed': 'write_retries_maxed_hotspotQA_to_file',
        'end': 'convert_ids_to_sentences'
    })

    graph.add_edge('convert_ids_to_sentences', END)

    app = graph.compile()

    answers_matched = 0
    supporting_facts_keys_matched = 0
    time_sum = timedelta(seconds=0)
    retry_sum = 0
    n = len(hotspotQA_train_data)
    observations = []
    print('running multi hop')
    for indx,train_data in enumerate(hotspotQA_train_data):
        start = time.perf_counter()
        state : HotSpotQA = {
            'question' : train_data['question'],
            'context': get_context(train_data['context']),
            'actual_answer' : train_data['answer'],
            'actual_supporting_facts': get_supporting_facts(train_data['supporting_facts'], train_data['context']),
            'prev_resp_comments' : [],
            'resp_answer': '',
            'resp_supporting_facts': {},
            'raw_response':'',
            'retry_count': 0,
            'pred_answer': '',
            'pred_supporting_facts' : {},
            'provide_answer': False,
            'question_id': train_data['id']
        }

        res = app.invoke(state)
        end = time.perf_counter()
        matched = False
        if res['pred_answer'].lower() == res['actual_answer'].lower():
            answers_matched+=1
            matched = True
        
        if res['pred_supporting_facts'].keys() == res['actual_supporting_facts'].keys():
            supporting_facts_keys_matched+=1
           
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
                'n' : n, 'exact_matched_answers' : answers_matched,
                'exact_matched_doc_keys' : supporting_facts_keys_matched,
                'avg_time_in_seconds' : time_sum.total_seconds() / n, 'retry_count' : retry_sum / n,
                'total_time_taken_in_seconds' : time_sum.total_seconds(),'total_retry_count' : retry_sum
            }
            }

    
    print(metrics)

    observations.append(metrics)
    with open(commons.MULTI_HOP_RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(observations, f, ensure_ascii=False, indent=2)

def run_single_hop_with_co_context():

    graph = StateGraph(HotSpotQA)
    graph.add_node('get_answer', query_llm_with_no_context)
    graph.add_node('write_retries_maxed_hotspotQA_to_file', write_retries_maxed_hotspotQA_to_file)
    graph.add_node('validate_answer', validate_answer)

    graph.set_entry_point('get_answer')
    graph.add_edge('get_answer', 'validate_answer')
    graph.add_edge('write_retries_maxed_hotspotQA_to_file', END)
   


    graph.add_conditional_edges('validate_answer', check_answer, {
        'retry': 'get_answer',
        'retry_maxed': 'write_retries_maxed_hotspotQA_to_file',
        'end': END
    })
    
    app = graph.compile()

    answers_matched = 0
    supporting_facts_keys_matched = 0
    time_sum = timedelta(seconds=0)
    retry_sum = 0
    n = len(hotspotQA_train_data)
    observations = []
    print('running single hop with no context')
    for indx,train_data in enumerate(hotspotQA_train_data):
        start = time.perf_counter()
        state : HotSpotQA = {
            'question' : train_data['question'],
            'context': {},
            'actual_answer' : train_data['answer'],
            'actual_supporting_facts': {},
            'prev_resp_comments' : [],
            'resp_answer': '',
            'resp_supporting_facts': {},
            'raw_response':'',
            'retry_count': 0,
            'pred_answer': '',
            'pred_supporting_facts' : {},
            'provide_answer': False,
            'question_id': train_data['id']
        }

        res = app.invoke(state)
        end = time.perf_counter()
        matched = False
        if res['pred_answer'].lower() == res['actual_answer'].lower():
            answers_matched+=1
            matched = True
        
        if res['pred_supporting_facts'].keys() == res['actual_supporting_facts'].keys():
            supporting_facts_keys_matched+=1

        
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
                'n' : n, 'exact_matched_answers' : answers_matched,
                'exact_matched_doc_keys' : supporting_facts_keys_matched,
                'avg_time_in_seconds' : time_sum.total_seconds() / n, 'retry_count' : retry_sum / n,
                'total_time_taken_in_seconds' : time_sum.total_seconds(),'total_retry_count' : retry_sum
            }
            }

    
    print(metrics)

    observations.append(metrics)
    with open(commons.SINGLE_HOP_WITH_NO_CONTEXT_RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(observations, f, ensure_ascii=False, indent=2)



api_key = os.getenv('OPEN-ROUTER-API-KEY')

llm = ChatOpenRouter(
  model="qwen/qwen3-32b",
  api_key=api_key, # type: ignore
  temperature=0
)

run_single_hop()


run_multi_hop()


run_single_hop_with_co_context()
