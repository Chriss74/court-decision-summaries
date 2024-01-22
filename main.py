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
    def __init__(self, model: str = 'gpt-3.5-turbo') -> None:
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
 
def openaicompletion(client: OpenAI, text, my_temperature, my_content, my_max=350):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=my_temperature,
        #max_tokens=my_max,
        messages=[
            {
                'role': 'system',
                'content': 'You are a legal professional. You will create summaries of court decisions in Greek using legal language'
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
    
def summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_content):
    merged_section_summary = ''
    for item in chunks:
        item_max_tokens=max(token_estimator.estimate(item)*section.importance, 350)
        response = openaicompletion(client, item, my_temperature, my_content, int(item_max_tokens))
        summary = response.choices[0].message.content
        merged_section_summary = f'{merged_section_summary} {summary}'
    return merged_section_summary
 
def main():
    load_dotenv()
 
    OPENAI_KEY=getenv('OPENAI_KEY')
    print(OPENAI_KEY)
 
    OPENAI_TOKEN_LIMIT = 4050
 
    annotation_mappings = read_json_file(join('documents', 'annotation_mappings.json'))
 
    annotated_decision = AnnotatedDecision.from_json(join('documents', 'annotated_decisions', 'ste_1537-2023.json'), annotation_mappings)
 
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
        
        if section.class_name=='other':
            continue
        elif section.class_name=='admissibility':
            my_temperature=0.7
            my_content="Summarize. If you can't, give the basic notion"
        elif section.class_name=='overview':
            my_temperature=0.8
            my_content="Summarize. If you can't, give the basic notion"
        elif section.class_name=='law':
            my_temperature=1.0
            my_content="Give the references of all the important laws (article/paragraphs included) in this text."
        elif section.class_name=='interpretation':
            my_temperature=0.5
            my_content="Mention the law that is interpreted and summarize"
        elif section.class_name=='previous-ruling':
            my_temperature=0.7
            my_content="Summarize. If you can't, give the basic notion"
        elif section.class_name=='facts':
            my_temperature=1.0
            my_content="Summarize the key facts"
        elif section.class_name=='party-claims':
            my_temperature=0.5
            my_content="Summarize. If you can't, give the basic notion"
        elif section.class_name=='court-response':
            my_temperature=0.5
            my_content="Summarize. If you can't, give the basic notion"
        elif section.class_name=='court-ruling':
            my_temperature=1
            my_content="Tell me what the court ruled"
        elif section.class_name=='important':
            my_temperature=0.5

        section_input_tokens = token_estimator.estimate(section.text)
        importance_times_tokens = section_input_tokens * section.importance
        portion = importance_times_tokens / all_importance_times_tokens
        section_summary_tokens = max (portion*summary_tokens, 400)
        print(section_summary_tokens)
        if section_summary_tokens + section_input_tokens +60> OPENAI_TOKEN_LIMIT:
            print(section.class_name, "was sliced")
            sentences = split_to_sentences(section.text)
            n_chunks = math.ceil((section_summary_tokens + section_input_tokens) / OPENAI_TOKEN_LIMIT)
            max_chunk_token_size = max((section_summary_tokens/ n_chunks) + (section_input_tokens / n_chunks)-60, 400)
            chunks=create_chunks(sentences,token_estimator, max_chunk_token_size)
            merged_section_summary=summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_content)
            print("if ", merged_section_summary)
            # chunks = []
            # chunk = ''
            # chunk_token_size = 0
            # for sentence in sentences:
            #     sentence_tokens = token_estimator.estimate(sentence)
            #     if chunk_token_size + sentence_tokens < max_chunk_token_size:
            #         chunk = f'{chunk} {sentence}'
            #         chunk_token_size = chunk_token_size + sentence_tokens
            #     else:
            #         chunks.append(chunk)
            #         chunk = sentence
            #         chunk_token_size = sentence_tokens
            # merged_section_summary = ''
            # for item in chunks:
            #     item_max_tokens=max(token_estimator.estimate(item)*section.importance, 350)
            #     response = openaicompletion(client, item, my_temperature, my_content, int(item_max_tokens))
            #     summary = response.choices[0].message.content
            #     merged_section_summary = f'{merged_section_summary} {summary}'
            if section.class_name=="law":
                law_list.append(merged_section_summary)
                continue
            merged_section_summary_tokens=token_estimator.estimate(merged_section_summary)
            while merged_section_summary_tokens>3300:
                sentences=split_to_sentences(merged_section_summary)
                chunks=create_chunks(sentences,token_estimator, 3300)
                merged_section_summary=summarize_summaries(chunks, token_estimator, section, client, my_temperature, my_content)
                print("while ", merged_section_summary)
                merged_section_summary_tokens=token_estimator.estimate(merged_section_summary)
            #TO DO: what if merged_section_summaries is too long? Chunks, same as before 
            # merged_section_summaries = openaicompletion(client, merged_section_summary, 1, "Summary this in Greek", int(section_summary_tokens))
            # summary_of_summaries = merged_section_summaries.choices[0].message.content
            # merged_full_summary=merged_full_summary+summary_of_summaries
            print(merged_section_summary)
            merged_full_summary=merged_full_summary+merged_section_summary
            
        else:
            response = openaicompletion(client, section.text, my_temperature, my_content, int(section_summary_tokens))
            summary = response.choices[0].message.content
            merged_full_summary=merged_full_summary+summary
            print(merged_full_summary)
            
    print(annotated_decision.document_id)
    print(annotated_decision.court)
    print(annotated_decision.related_department)
    print(annotated_decision.legal_remedy)
    print("ΣΗΜΑΝΤΙΚΗ ΝΟΜΟΘΕΣΙΑ")
    print(law_list)
    print ("ΠΕΡΙΛΗΨΗ")
    print(merged_full_summary)
            
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