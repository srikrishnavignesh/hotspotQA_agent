import json

context = {'title a':{'0': 'sentence'}, 'title b':{'0': 'sentence'}}
input = {'question':'', 'context':context, 'prev_resp_comments':'', 'provide_answer': ''}
output = {'answer':'', 'supporting_facts':{'title a':[], 'title b':[]}, "context_needed": []}


BASE_PROMPT = f"""
You are a precise question-answering assistant.
INPUT_FORMAT:
Input is the following format {json.dumps(input, ensure_ascii=False)}.
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
    only focus on providing supporting facts.

OUTPUT:
output must be strictly in the following format {json.dumps(output, ensure_ascii=False)}.
'answer':
    - Do not provide explanations. Only provide the answer. No conversations. No thinking out loud.
    - If the answer is 'yes' or 'no', return exactly that.
        Otherwise use the verbatim from the context that answers the question.
    - If the answer is derieved from multiple sentences in the context, 
        you can combine the relevant sentences verbatim to form the answer.
    - If context is not sufficient to come up with an answer.
      return 'Insufficient Data'
'supporting_facts':
    -The keys are the titles from the context that leads and progresses to the answer.
    -The values are those corresponding titles sentence ids.
    -You must strictly only include titles and sentence ids from the 'context' that lead to the
        answer. You must not include random or arbitrary titles and sentence ids.
        Do not invent titles and sentence ids.
    -You must provide all the titles and sentence ids that are needed to reason and answer the question.
     Do not leave any.
    -Looking just at the facts we must be able to answer the question with zero doubtfulness.
'context_needed':
    - You must use this field to identify any missing information that is needed to answer the question 
        when the provided context is insufficient.
    - Provide a list of precise and complete Wikipedia-style titles required to answer the question.
    - Each title must be:
        • Fully qualified and unambiguous (include disambiguation if needed, e.g., "Titanic (1997 film)")
    - Do NOT provide generic queries or partial phrases  (e.g., avoid "Titanic", "war history") and 
      prefer exact entity names(person, place, event, film, book, etc.)
    - The titles should be directly usable for retrieval (e.g., BM25 search).
    - If sufficient information is already present in the context, return an empty list.
    - When 'provide_answer' is true, this field must be an empty list.
    - You can provide atmost 3 titles in this field.


-REASONING:
    -Reasoning is internal only. Do not include it in output.
    -Identify and eliminate all distractor sentences in the 'context'.
    -Restrict reasoning to relevant titles and sentences only.
    -Perform answer derivation on filtered context.
    -You must derive the answer from multiple sentences of different 'titles', the answer may not be derivable 
    from just one single sentence, So you must focus on connecting relevant sentences from multiple titles 
    to answer the question.
    -Connecting the relevant sentences and removing the noise is the most critical part.
    -If 'answer' is True and 'context' is not sufficient to give answer, provide 'insufficient context' in the answer.
-Titles already present in the 'context' must not be repeated in 'context_needed'.
"""



def get_query_prompt_with_context(question, context, prev_resp_comments, provide_answer):
    input = {'question': question, 'context': context, 'prev_resp_comments': prev_resp_comments, 
             'provide_answer': provide_answer}
    return json.dumps(input, ensure_ascii=False)

INVALID_JSON_FORMAT_PROMPT = 'Response is not a valid json. Please ensure your response strictly follows the output format.'

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

TOP_K_RETRIEVAL = 7


TEST_DATA_PATH = 'test.json'

RETRIES_MAXED_FILE_PATH = 'pyserini/retries_maxed_hotspotQA.json'

MAX_RETRIES = 5

PYSERSINI_BASED_QUERY_RESULTS_FILE_PATH = 'pyserini/pyserini_based_query_results.json'

MAX_ITERATIONS = 3

PYSERINI_DOCS_FILE_LOC = f'pyserini/docs'
PYSERINI_INDEX_FILE_DOX = f'pyserini/index'

PYSERINI_DOCS = 'pyserini/docs'

PYSERINI_FILES_PREFIX = 'doc_'


INPUT_DATA = 'input.json'