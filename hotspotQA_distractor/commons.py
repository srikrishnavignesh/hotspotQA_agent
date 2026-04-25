
import json
context = {'title a':{'0': 'sentence'}, 'title b':{'0': 'sentence'}}
input = {'question':'', 'context':context, 'prev_resp_comments':'', 'provide_answer': ''}
input_without_context = {'question':'', 'prev_resp_comments':''}


output = {'answer':'', 'supporting_facts':{'title a':[], 'title b':[]}}
output_without_supporting_facts = {'answer':''}

BASE_PROMPT = f"""
You are a precise question-answering assistant.
                    
INPUT_FORMAT:
Input is the following format {json.dumps(input)}.
'question:'
    We have a single question.
'context':
    We also have a context in the following format {json.dumps(context)}.
    Keys are the different titles.
    values are key value pairs, where
        - The keys represent the sentence ids.
        - The value represent the actual sentence.
'prev_resp_comments':
    - If 'prev_resp_comments' is not empty, it means prev iteration had some errors.
    - Identify what was wrong
    - Correct your answer accordingly
    - Do not repeat the same mistake 
'provide_answer':
    - If 'provide_answer' is true, it means you should provide an answer in this iteration.
    - If 'provide_answer' is false, it means you should not provide an answer in this iteration and 
    only focus on providing supporting facts.

REASONING:
-You must reason with only  the provided 'context'.
-You must identify and combine information from multiple sentence if needed.
-You must Analyze all the titles and sentences in the context before answering. 
-Do not answer based on partial analysis.
OUTPUT:
output must be strictly in the following format {json.dumps(output)}.
'answer':
    - Do not provide explanations. Only provide the answer. No conversations. No thinking out loud.
    - Do not provide an answer if 'provide_answer' is 'false'. Return empty string if 'false'. 
        You must strictly honour the 'provide_answer' field.
    - If the answer is 'yes' or 'no', return exactly that.
        Otherwise use the verbatim from the context that answers the question.
    - If the answer is derieved from multiple sentences in the context, 
        you can combine the relevant sentences verbatim to form the answer. 
    - Do not leave this field empty when 'provide_answer' is 'true'. 
        You must provide an answer when 'provide_answer' is 'true'. 
'supporting_facts':
    -The keys are the titles from the context that lead to the answer.
    -The values are those corresponding titles sentence ids.
    -You must strictly only include titles and sentence ids from the 'context' that lead to the
        answer. You must not include random or arbitary titles and sentence ids.
        Do not invent titles and sentence ids.
    -You must provide  all the titles and sentence ids that are needed to reason and answer the question.
        Do not leave any.

No conversations. No thinking out loud.
Internally reason step-by-step, but do not output reasoning.
Only output json.
"""

NO_CONTEXT_PROMPT = f"""
INPUT_FORMAT:
The input is in the following format  {json.dumps(input_without_context)}. 
'question:'
    -We have a single question.
    -You have to provide anwer for the question based on your internal knowledge and reasoning.
'prev_resp_comments':
    - If 'prev_resp_comments' is not empty, it means prev iteration had some errors.
    - Identify what was wrong
    - Correct your answer accordingly
    - Do not repeat the same mistake 
'Response:'
    -The response must be exactly in the following JSON format {json.dumps(output_without_supporting_facts)}.
    -Ouput must be a valid JSON.
    -You must provide exactly one answer.
    -No conversation. No thinking out loud. Do not output explanations or reasonings.
    -Just provide the answer alone. Nothing more. Nothing less.
    - If the answer is 'yes' or 'no', return exactly 'yes' or 'no'.
    -Internally reason step-by-step, but do not output reasoning.
    -You must come up with the answer based on your internal knowledge and reasoning. 
    -Do not use any external information or context as there is none provided.
"""

def get_query_prompt_with_context(question, context, prev_resp_comments, provide_answer):
    input = {'question':question, 'context':context, 'prev_resp_comments':prev_resp_comments, 'provide_answer': provide_answer}
    return f'{json.dumps(input, indent=2)}.'

def get_query_prompt_without_context(question, prev_resp_comments):
    input = {'question':question, 'prev_resp_comments': prev_resp_comments}
    return f'{json.dumps(input, indent=2)}.'

def get_invalid_sentence_ids_prompt(title, invalid_sentence_ids):
 return f"""
            The sentence ids provided for the {title} contains the following invalid sentence ids {invalid_sentence_ids},
            You must ensure that the sentence ids or not invented and must provide valid reasoning for the answers.
        
        """

def get_invalid_titles_prompt(title):
   return f"""
            The title provided in 'supporting_facts' is invalid {title},
            You must ensure that the titles or not invented and must provide valid reasoning for the answers
          """

def get_no_answer_in_respone_prompt():
   return f"""
            The response is missing the 'answer' field. You must ensure that your response has a valid 'answer' if the
            'provide_answer' field is True. You must honour this strictly.
          """

INVALID_JSON_FORMAT_PROMPT = 'Response is not a valid json. Please ensure your response strictly follows the output format.'

MAX_RETRIES = 5

BASE_PATH = 'hotspotQA_distractor'

RETRIES_MAXED_FILE_PATH = f'{BASE_PATH}/retries_maxed_hotspotQA.json'

SINGLE_HOP_RESULTS_FILE_PATH = f'{BASE_PATH}/single_hop_results.json'

MULTI_HOP_RESULTS_FILE_PATH = f'{BASE_PATH}/multi_hop_results.json'

SINGLE_HOP_WITH_NO_CONTEXT_RESULTS_FILE_PATH = f'{BASE_PATH}/single_hop_with_no_context_results.json'

TRAIN_DATA_PATH = f'{BASE_PATH}/train.json'

