import json
import re
import math
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from os import getenv, makedirs
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


def split_text_into_chunks(text, max_tokens, token_estimator):
    """Σπάει το κείμενο σε τμήματα που δεν ξεπερνούν τα max_tokens."""
    sentences = text.split(". ")
    chunks, chunk, chunk_token_size = [], "", 0

    for sentence in sentences:
        sentence_tokens = token_estimator.estimate(sentence)
        if chunk_token_size + sentence_tokens > max_tokens:
            chunks.append(chunk.strip())
            chunk, chunk_token_size = sentence + ". ", sentence_tokens
        else:
            chunk += sentence + ". "
            chunk_token_size += sentence_tokens

    if chunk:
        chunks.append(chunk.strip())
    return chunks


def openai_completion(client: OpenAI, text, temperature, prompt, content, max_tokens=400):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'{content}:{text}'}
        ],
    )
    return response


def main():
    load_dotenv(override=True)
    OPENAI_KEY = getenv('OPENAI_KEY')
    OPENAI_TOKEN_LIMIT = 16385  # Μέγιστο όριο tokens για το μοντέλο
    SAFE_TOKEN_LIMIT = 10000  # Κρατάω περιθώριο για prompt
    my_prompt = (
        "You are a legal expert specializing in public procurement law. Provide an analytical summary focusing on the interpretation of legal principles, avoiding direct citations of legal texts unless explicitly important. Emphasize theoretical aspects and implications rather than procedural details. The summary should be in Greek."
    )
    annotation_mappings = read_json_file(join('documents', 'annotation_mappings.json'))
    decision_number = 'ste_2325-2023'
    annotated_decision = read_json_file(join('documents', 'annotated_decisions', f'{decision_number}.json'))
    
    client = OpenAI(api_key=OPENAI_KEY)
    token_estimator = TokenEstimator()
    merged_full_summary = ''
    
    for idx, section in enumerate(annotated_decision['annotations'], start=1):
        class_name = section.get('name', 'other')
        total_parts = len(annotated_decision['annotations'])
        print(f'Μέρος {idx} από {total_parts}: {class_name}')
        max_section_summary_tokens = 700
        my_temperature, my_content = 0.4, 'Summarize the interpretation of the following legal text in Greek, avoiding direct citations unless they are critical.'
        #πολλά tokens=περισσότερη πληροφορία χαμηλό temperature=ακριβέστερη απόδοση
        if class_name == 'law' or class_name == 'important':
            my_content = "Provide an interpretation of the legal provisions, including key references only if they are critical. The summary should be in Greek."
            my_temperature, my_content, max_section_summary_tokens = (
                0.3, 'Provide a structured interpretation of the legal provisions mentioned, focusing on their implications and theoretical aspects. The summary should be in Greek.', 1000
            )
        elif class_name in ['admissibility', 'interpretation', 'previous-ruling', 'court-response']:
            my_temperature = 0.4
        elif class_name == 'overview':
            my_temperature = 0.6
        elif class_name in ['facts', 'court-ruling', 'court-response']:
            max_section_summary_tokens = 900  
            my_temperature = 0.5  
            max_section_summary_tokens = 500  
            my_temperature = 0.5
            my_content = 'Summarize the key legal aspects without excessive case details and avoid direct legal citations unless necessary. The summary should be in Greek.'
        elif class_name == 'party-claims':
            my_temperature = 0.4
        elif class_name == 'important':
            my_temperature = 0.5 
            my_temperature = 0.7
            my_content = 'Provide a deeper interpretation of this section, including references to legal texts only if they are essential. The summary should be in Greek.'
        
        response = openai_completion(
            client, section['text'], my_temperature, my_prompt, my_content, int(max_section_summary_tokens)
        )
        merged_section_summary = response.choices[0].message.content
        merged_full_summary += f' {merged_section_summary}'
    
    # Περιορισμός μεγέθους κειμένου για αποφυγή OpenAI error
    merged_summary_tokens = token_estimator.estimate(merged_full_summary)
    if merged_summary_tokens > SAFE_TOKEN_LIMIT:
        print(f'Προειδοποίηση: Το σύνολο της περίληψης ({merged_summary_tokens} tokens) είναι πολύ μεγάλο. Μειώνεται για αποφυγή σφάλματος.')
        merged_full_summary = ' '.join(merged_full_summary.split()[:SAFE_TOKEN_LIMIT]) if merged_summary_tokens > SAFE_TOKEN_LIMIT else merged_full_summary
    
    # Αν το merged_full_summary είναι πολύ μεγάλο, το χωρίζω σε μικρότερα chunks
    if merged_summary_tokens > SAFE_TOKEN_LIMIT:
        print(f'Προειδοποίηση: Το σύνολο της περίληψης ({merged_summary_tokens} tokens) είναι πολύ μεγάλο. Χωρίζεται σε μικρότερα τμήματα για αποφυγή σφάλματος.')
        chunks = split_text_into_chunks(merged_full_summary, 4000, token_estimator)
        summarized_chunks = []
        
        for chunk in chunks:
            chunk_summary = openai_completion(
                client, chunk, 0.4, my_prompt,
                "Summarize this section concisely in Greek while keeping key legal principles.",
                800
            )
            summarized_chunks.append(chunk_summary.choices[0].message.content)
        
        merged_full_summary = ' '.join(summarized_chunks)
    
    # Εισαγωγή γενικού συμπεράσματος
    summary_conclusion = openai_completion(
        client, merged_full_summary, 0.4, my_prompt,
        "Summarize the key legal principles and their broader implications, excluding procedural references unless necessary. The summary should be in Greek.",
        800
    )
    merged_full_summary += f"\n\nΣυμπέρασμα:\n{summary_conclusion.choices[0].message.content}"
    
    output_folder = 'documents/my_summaries'
    makedirs(output_folder, exist_ok=True)
    with open(join(output_folder, f'{decision_number}.txt'), 'w', encoding='utf-8') as file:
        file.write(merged_full_summary.strip())
    
    print(f'Περίληψη αποθηκεύτηκε: {decision_number}.txt')


if __name__ == '__main__':
    main()
