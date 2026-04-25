import json
import re
from collections import Counter

def get_word_freq(line):
    word_freq = Counter()
    for word in line:
       word_freq[word]+=1
    return word_freq

def clean(answer):
    answer = re.sub(r'\s+', ' ', answer).lower()
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer.strip()
    
def get_f1_for_answer(pred_answer, actual_answer):
    pred_answer_cleaned = clean(pred_answer)
    
    if pred_answer == '':
          return 0
     
    if pred_answer == actual_answer:
            return 1
    
    actual_answer_cleaned = clean(actual_answer)
    pred_answer_words = pred_answer_cleaned.split(' ')
    actual_answer_words = actual_answer_cleaned.split(' ')

    actual_w_f = get_word_freq(actual_answer_words)
    pred_w_f = get_word_freq(pred_answer_words)

    tp = 0
    ap = 0
    for w, actual_freq in actual_w_f.items():
        if w in pred_w_f:
            tp+= min(actual_freq, pred_w_f[w])
        ap+=actual_freq
    
    pp = 0
    for w, actual_freq in pred_w_f.items():
        pp+=actual_freq

    recall = tp/ap
    precision = tp/pp 
    
    if precision == 0 and recall == 0:
         return 0
    
    return 2*(precision*recall)/(precision+recall)

def get_f1_for_supporting_facts(pred_sf, actual_sf):
    if len(pred_sf) == 0 or len(actual_sf) == 0:
          return 0
     
    if pred_sf == actual_sf:
            return 1
     
    tp = 0
    ap = 0
    for k, actual_sent_ids in actual_sf.items():
        set_actual_sent_ids = set(actual_sent_ids)
        if k in pred_sf:
            set_pred_sent_ids = set(pred_sf[k])
            inter = (set_pred_sent_ids).intersection(set_actual_sent_ids)
            tp+=len(inter)
        ap+=len(set_actual_sent_ids)
    
    pp = 0
    for k, pred_sent_ids in pred_sf.items():
        set_pred_sent_ids = set(pred_sent_ids)
        pp+=len(set_pred_sent_ids)

    recall = tp/ap

    if pp == 0:
         return 0
    
    precision = tp/pp

    if precision == 0 and recall == 0:
         return 0

    return 2*(precision*recall)/(precision+recall)

def answers_exact_match(pred_answer, actual_answer):
     return clean(pred_answer) == clean(actual_answer)


def get_score(hotspotQA_results):
    answers_f1_sum = 0
    answers_em_sum = 0
    supporting_facts_em_sum = 0
    supporting_facts_f1_sum = 0

    n = len(hotspotQA_results)
    for hotspotQA_result in hotspotQA_results[:-1]:
        pred_answer, actual_answer = hotspotQA_result['pred_answer'],hotspotQA_result['actual_answer']
        
        answer_f1 = get_f1_for_answer(pred_answer, actual_answer)
        answers_em_sum+= 1 if answers_exact_match(pred_answer, actual_answer) else 0 
        answers_f1_sum+=answer_f1

        pred_supporting_facts, actual_supporting_facts = hotspotQA_result['pred_supporting_facts'], hotspotQA_result['actual_supporting_facts']
        supporting_facts_f1 = get_f1_for_supporting_facts(pred_supporting_facts, actual_supporting_facts)
        
        supporting_facts_em_sum+= 1 if supporting_facts_f1 == 1 else 0
        supporting_facts_f1_sum+=supporting_facts_f1
    
    return answers_em_sum/n, answers_f1_sum/n, supporting_facts_em_sum/n, supporting_facts_f1_sum/n


SINGLE_HOP_NO_CONTEXT = 'hotspotQA_distractor/single_hop_with_no_context_results.json'

MULTI_HOP_WITH_CONTEXT = 'hotspotQA_distractor/multi_hop_results.json'

SINGLE_HOP_WITH_CONTEXT = 'hotspotQA_distractor/single_hop_results.json'

PYSERINI_CONTEXT_WITH_MULTI_HOP = 'hotspotQA_fullwiki/pyserini_based_query_results.json'

files = {'single_hop_no_context': SINGLE_HOP_NO_CONTEXT, 
         'single_hop_with_context':SINGLE_HOP_WITH_CONTEXT, 
         'multi_hop_refinement': MULTI_HOP_WITH_CONTEXT, 
         'pyserini_based_multi_hop': PYSERINI_CONTEXT_WITH_MULTI_HOP
         }

for method, file in files.items():
    with open(file, 'r', encoding='utf-8') as f:
        answer_em, answer_f1,supporting_facts_em, supporting_facts_f1 =  get_score(json.load(f))
        print(f"""method:{method} answer_em : {answer_em}, answer_f1 : {answer_f1},  supporting_facts_em : {supporting_facts_em}', 
              supporting_facts_f1 : {supporting_facts_f1}""")
        




            


            

