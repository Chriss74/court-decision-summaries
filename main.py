import json
import re
import math
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
from typing import List, Dict, Any
from os.path import join
 
 
def read_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
 
class TokenEstimator:
    def __init__(self, model: str = 'gpt-4') -> None:
        self.encoding = tiktoken.encoding_for_model(model)
 
    def estimate(self, text: str) -> int:
        return len(self.encoding.encode(text))
 
class AnnotatedSection:
    def __init__(
            self, 
            class_id: List[str], 
            text: str,
            importance: int,
            class_name: str) -> None:
        self.class_id = class_id
        self.text = text
        self.importance = importance
        self.class_name = class_name
        
    
        
 
class AnnotatedDecision:
    def __init__(
            self, 
            document_id: str, 
            court: str, 
            legal_remedy: str, 
            related_department: str,
            annotations: List[AnnotatedSection]) -> None:
        
        self.document_id = document_id
        self.court = court
        self.legal_remedy = legal_remedy
        self.related_department = related_department
        self.annotations = annotations
    
    @classmethod
    def from_json(cls, annotated_decision_path: str, annotated_mappings_dict: Dict[str, Any]) -> 'AnnotatedDecision':
        annotated_decision_dict = read_json_file(annotated_decision_path)
 
        annotated_sections = [
            AnnotatedSection(
                section['class_id'], 
                section['text'], 
                annotated_mappings_dict[section['class_id']]['importance'],
                annotated_mappings_dict[section['class_id']]['name'])
            for section in annotated_decision_dict['annotations']]
 
        return AnnotatedDecision(
            annotated_decision_dict['document_id'],
            annotated_decision_dict['court'],
            annotated_decision_dict['legal_remedy'],
            annotated_decision_dict['related_department'],
            annotated_sections
            )
    
    def construct_document(self):
        document = ''
        for annotated_section in self.annotations:
            document = f'{document} {annotated_section.text}'
 
        return document
    
    def get_document_tokens(self, token_estimator: TokenEstimator) -> int:
        return token_estimator.estimate(self.construct_document())
    
    
def split_to_sentences(text: str) -> List[str]:
    sentences = []
    previous_fullstop_index = 0
 
    for match in re.finditer(r'(\.)(\s+[Α-Ω])', text):
        fullstop_index = match.span()[0]
        sentences.append(text[previous_fullstop_index+1:fullstop_index+1])
        previous_fullstop_index = fullstop_index
    
    return sentences
 
def openaicompletion(client: OpenAI, text, my_temperature, my_prompt, my_content, my_max=400):
    print(my_prompt)
    print(my_content)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=my_temperature,
        #max_tokens=my_max,
        messages=[
            {
                'role': 'system',
                'content': f'{my_prompt}'
            },
            {
                'role': 'user',
                'content': f'{my_content}:\n{text}'
                
            }
        ],
    )
 
    return response

def create_chunks(sentences,token_estimator, max_chunk_token_size):
    chunks = []
    chunk = ''
    chunk_token_size = 0
    for sentence in sentences:
        sentence_tokens = token_estimator.estimate(sentence)
        if chunk_token_size + sentence_tokens < max_chunk_token_size:
            chunk = f'{chunk} {sentence}'
            chunk_token_size = chunk_token_size + sentence_tokens
        else:
            chunks.append(chunk)
            chunk = sentence
            chunk_token_size = sentence_tokens
    return chunks
    
def summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_prompt, my_content):
    merged_section_summary = ''
    for item in chunks:
        item_max_tokens=max(token_estimator.estimate(item)*section.importance, 400)
        response = openaicompletion(client, item, my_temperature, my_prompt, my_content, int(item_max_tokens))
        summary = response.choices[0].message.content
        merged_section_summary = f'{merged_section_summary} {summary}'
    return merged_section_summary


            
    
# def find_max_x():
#     max_x = 1  # Initialize max_x to the smallest possible positive integer

#     for x in range(2700, 4050 + 1):
#         y = a / x
#         if y.is_integer() and x > max_x:
#             max_x = x

#     return max_x
 
def main():
    load_dotenv()
 
    # OPENAI_KEY=getenv('OPENAI_KEY')
    OPENAI_KEY="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"
    print(OPENAI_KEY)
 
    OPENAI_TOKEN_LIMIT = 4050
    my_prompt='You are a legal professional. Answer in greek and use legal language'
    
    annotation_mappings = read_json_file(join('documents', 'annotation_mappings.json'))
 
    annotated_decision = AnnotatedDecision.from_json(join('documents', 'annotated_decisions', 'ste_2325-2023.json'), annotation_mappings)
 
    print(annotated_decision.document_id)
 
    client = OpenAI(api_key=OPENAI_KEY)
    
    token_estimator = TokenEstimator()
 
    decision_tokens = annotated_decision.get_document_tokens(token_estimator)
 
    summary_tokens = max(decision_tokens * 0.3, 15000)
 
    all_importance_times_tokens = sum([(annotation.importance * token_estimator.estimate(annotation.text)) for annotation in annotated_decision.annotations])
    
    print("all_importance_times_tokens=",all_importance_times_tokens)
    law_list=[]
    merged_full_summary=''
    for section in annotated_decision.annotations:
        print(section.class_name)
        max_section_summary_tokens=400
        if section.class_name=='other':
            continue
        elif section.class_name=='admissibility':
            my_temperature=0.7
            my_content="Summarize. If you can't, give the basic notion."
        elif section.class_name=='overview':
            my_temperature=0.8
            my_content="Summarize. If you can't, give the basic notion."
        elif section.class_name=='law':
            my_temperature=1.0
            my_content="Give a list of number article, paragraph, name and the title of the law/article, mentioned in this text"
            max_section_summary_tokens=250
        elif section.class_name=='interpretation':
            my_temperature=0.7
            my_content="Mention the law that is interpreted and summarize."
        elif section.class_name=='previous-ruling':
            my_temperature=0.7
            my_content="Summarize. If you can't, give the basic notion."
        elif section.class_name=='facts':
            my_temperature=1.0
            my_content="Summarize the key facts."
        elif section.class_name=='party-claims':
            my_temperature=0.5
            my_content="Summarize. If you can't, give the basic notion.Use greek language"
        elif section.class_name=='court-response':
            my_temperature=0.7
            my_content="Summarize. If you can't, give the basic notion.Use greek language"
        elif section.class_name=='court-ruling':
            my_temperature=1
            my_content="Tell me what the court ruled.Use greek language"
        elif section.class_name=='important':
            my_temperature=0.7

        section_input_tokens = token_estimator.estimate(section.text)
        importance_times_tokens = section_input_tokens * section.importance
        portion = importance_times_tokens / all_importance_times_tokens
        section_summary_tokens = max (portion*summary_tokens, max_section_summary_tokens)
        all_prompt_tokens=math.floor(token_estimator.estimate(my_content)+token_estimator.estimate(my_prompt))
        print(section_summary_tokens)
        if section_summary_tokens + section_input_tokens + all_prompt_tokens> OPENAI_TOKEN_LIMIT:
            print(section.class_name, "was sliced")
            sentences = split_to_sentences(section.text)
            n_chunks = math.ceil((section_summary_tokens + section_input_tokens) / OPENAI_TOKEN_LIMIT)
            chunk_token_size=min((section_summary_tokens/ n_chunks) + (section_input_tokens / n_chunks)-all_prompt_tokens,3200)
            max_chunk_token_size = math.floor(max(chunk_token_size, max_section_summary_tokens))
            #Math nx=4050/1+portion 
            chunks=create_chunks(sentences,token_estimator, max_chunk_token_size)
            merged_section_summary=summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_prompt, my_content)
            if section.class_name=="law":
                law_list.append(merged_section_summary)
                continue
            merged_section_summary_tokens=token_estimator.estimate(merged_section_summary)
            while merged_section_summary_tokens>3200:
                sentences=split_to_sentences(merged_section_summary)
                chunks=create_chunks(sentences,token_estimator, 3300)
                merged_section_summary=summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_prompt, my_content)
                print("while ", merged_section_summary)
                merged_section_summary_tokens=token_estimator.estimate(merged_section_summary)
            #TO DO: what if merged_section_summaries is too long? Chunks, same as before DONE!!!!!
            #TO DO: calculate token length of prompt and use it to calculate chunk sizes
            merged_full_summary=merged_full_summary+merged_section_summary
            
        else:
            response = openaicompletion(client, section.text, my_temperature, my_prompt, my_content, int(section_summary_tokens))
            summary = response.choices[0].message.content
            merged_full_summary=merged_full_summary+summary
            
    merged_full_summary_tokens=token_estimator.estimate(merged_full_summary)
    if merged_full_summary_tokens>10000:
         sections = [merged_full_summary[i:i + 2500] for i in range(0, len(merged_full_summary), 2500)]
    my_content="Further summarize this court decision summary"
    summaries = []
    for section in sections:
        response = openaicompletion(client, merged_full_summary, my_temperature, my_prompt, my_content,1500)
        summary = response.choices[0].message.content
        summaries.append(summary)
    
    print(annotated_decision.document_id)
    print(annotated_decision.court)
    print(annotated_decision.related_department)
    print(annotated_decision.legal_remedy)
    print("ΣΗΜΑΝΤΙΚΗ ΝΟΜΟΘΕΣΙΑ")
    print(law_list)
    print ("ΠΕΡΙΛΗΨΗ")
    print("\n".join(summaries))
            
            #print(chunks)
            # for item in chunks:
            #     print(item)
            #     print(token_estimator.estimate(item))
            #     print(max_chunk_token_size)
            #     print('\n\n')
 
        #response = openaicompletion(client, section.text)
        #summary = response.choices[0].message.content
 
 
 
    # for item in query_parser.split_to_sentences():
    #     print(item)
    #     print('\n\n\n')
    
 
main()