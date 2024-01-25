import openai
import json
from typing import Dict, Any
from os.path import join

def read_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
    
class Decision:
    def __init__(
            self, 
            document_id: str, 
            court: str, 
            legal_remedy: str, 
            related_department: str,
            text: str) -> None:
        self.document_id = document_id
        self.court = court
        self.legal_remedy = legal_remedy
        self.related_department = related_department,
        self.text = text
        
    def from_json(cls, annotated_decision_path: str) -> 'Decision':
        annotated_decision_dict = read_json_file(annotated_decision_path)

        document_id = annotated_decision_dict.get('document_id', '')
        court = annotated_decision_dict.get('court', '')
        legal_remedy = annotated_decision_dict.get('legal_remedy', '')
        related_department = annotated_decision_dict.get('related_department', '')
        annotations = annotated_decision_dict.get('annotations', [])
        text = annotated_decision_dict.get('text', '')

        return Decision(document_id, court, legal_remedy, related_department, annotations,text)
    

def summarize_long_text(text, max_tokens=4096, section_length=3300):
    
    sections = [text[i:i + section_length] for i in range(0, len(text), section_length)]

    summaries = []
    for section in sections:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal professional. I will give you a part of a court decision. Answer in greek and use legal language"},
                {"role": "user", "content": section}
            ],
            max_tokens=min(max_tokens, 4050 - len(section)),  # Limit max_tokens per section
            temperature=0.7  # You can adjust the temperature based on your preference
        )
        summary = response.choices[0].message.content
        summaries.append(summary)

    return "\n".join(summaries)

   
# Set your OpenAI API key
openai.api_key = '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'
file_path=join('documents', 'annotated_decisions', 'ste_2325-2023.json')
# Generate the summarized text
decision=Decision.from_json(file_path)

result = summarize_long_text(decision.text)

# Print the summarized text
print(result)