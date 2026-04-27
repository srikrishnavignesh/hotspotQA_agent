import json


context = {'title a':{'0': 'sentence'}, 'title b':{'0': 'sentence'}}
input = {'question':'', 'context':context, 'prev_resp_comments':'', 'provide_answer': ''}
output = {'answer':'', 'supporting_facts':{'title a':[], 'title b':[]}, "context_needed": []}


BASE_PROMPT = f"""
You are a precise question-answering assistant.
INPUT_FORMAT:
Input is in the following format {json.dumps(input, ensure_ascii=False)}.
'question:'
    We have a single question.
'context':
    -We also have a context in the following format {json.dumps(context, ensure_ascii=False)}.
    -Keys are the different titles.
    -values are key value pairs, where
        - The keys represent the sentence ids.
        - The value represent the actual sentence.
    -The context includes distractor sentences as well.
'prev_resp_comments':
    - If 'prev_resp_comments' is not empty, it means prev iteration had some errors.
    - Identify what was wrong
    - Correct your answer accordingly
    - Do not repeat the same mistake 
'provide_answer':
    - If 'provide_answer' is true, it means you should provide an answer in this iteration.
    - If 'provide_answer' is false, it means you should not provide an answer in this iteration and 
    only focus on providing 'supporting_facts'.

OUTPUT:
output must be strictly in the following format {json.dumps(output, ensure_ascii=False)}.
'answer':
    - Do not provide explanations. Only provide the answer. No conversations. No thinking out loud.
      No introductory text or conversations fillers.
    - If the answer is 'yes' or 'no', return exactly that.
        Otherwise use the verbatim from the context that answers the question.
    - If the answer is derieved from multiple sentences in the context, 
        you can combine the relevant sentences verbatim to form the answer.
    - If context is not sufficient to come up with an answer, return 'insufficient context'
'supporting_facts':
    -The keys are the titles from the context that leads and progresses to the answer.
    -The values are those corresponding titles sentence ids.
    -You must strictly only include titles and sentence ids from the 'context' that lead to the
     answer. You must not include random or arbitrary titles and sentence ids.
     Do not invent titles and sentence ids.
    -You must provide all the titles and sentence ids that are needed to reason and answer the question.
     Do not leave any.
'context_needed':
    - You must use this field to identify any missing information that is needed to answer the question 
      when the provided context is insufficient.
    - You must come up with context rich queries to derieve the answer.
    - The queries will be used to retrieve data from the vector db.
    - You can provide atmost 3 queries in this field.
    - Avoid generic queries 
     
     'Bad Queries'-
     'Who is Donald Trump?' or 'When was Einstein born?' or '26/11 Mumbai Terror Attack'.

     'Good Queries'-
     'Donald Trump the president of united states of America'.
     'when was Einstein the famous physict, who introduced the ideas of general relativity and Mass-Energy equivalence born?'.
     'Terror Attack in Mumbai,India on september 26 2008 on places like Nariman House, Taj Hotel by 
      Lashkar-e-Taiba (terrorist organization).'

    You must never provide existing context or given question as part of the 'context_needed'.
      
      Adding better relevant contexts leads to better queries.

    - Return empty context if provided context is sufficient to asnwer the question.


-REASONING:
    -reasoning is internal only. Do not include it in output.
    -Identify and eliminate all distractor sentences in the 'context'.
    -Restrict reasoning to relevant titles and sentences only.
    -Perform answer derivation on filtered context.
    -You must derive the answer from multiple sentences of different 'titles', the answer may not be derivable 
    from just one single sentence, So you must focus on connecting relevant sentences from multiple titles 
    to answer the question.
    -Connecting the relevant sentences and removing the noise is the most critical part.
"""


def get_query_prompt_with_context(question, context, prev_resp_comments, provide_answer):
    input = {'question': question, 'context': context, 'prev_resp_comments': prev_resp_comments, 
             'provide_answer': provide_answer}
    return json.dumps(input, ensure_ascii=False)


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

def get_no_supporting_facts_provided_prompt():
   return "No supporting facts provided , You must make sure to provide valid titles " \
          "and sentence_ids that will lead to the answer from the provided 'context' "

def get_no_answer_in_respone_prompt():
   return f"""
            The response is missing the 'answer' field. You must ensure that your response has a valid 'answer' if the
            'provide_answer' field is True. You must honour this strictly.
            """




CHROMADB_PATH = 'sentence_transformers/chromadb'
CHROMADB_COLLECTION_NAME = 'hotspotQA'

TEST_DATA_PATH = 'test.json'

SENTENCE_TRANSFORMERS_BASED_RESULTS = 'sentence_transformers/sentence_transformers_based_results.json'

TOP_K_RETRIEVAL = 7


RETRIES_MAXED_FILE_PATH = 'sentence_transformers/retries_maxed_hotspotQA.json'

MAX_RETRIES = 5


MAX_ITERATIONS = 3


INPUT_DATA = 'input.json'

INVALID_JSON_FORMAT_PROMPT = 'Response is not a valid json. Please ensure your response strictly follows the output format.'

