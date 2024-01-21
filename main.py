import openai
import json
from dotenv import load_dotenv
from os import getenv
from typing import List, Dict, Any
from os.path import join
 
 
def read_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path) as json_file:
        return json.load(json_file)
 
class AnnotedSection:
    def __init__(
            self, 
            class_id: List[str], 
            text: str) -> None:
        self.class_id = class_id
        self.text = text
 
class AnnotatedDecision:
    def __init__(
            self, 
            document_id: str, 
            court: str, 
            legal_remedy: str, 
            related_department: str,
            annotations: List[AnnotedSection]) -> None:
        
        self.document_id = document_id
        self.court = court
        self.legal_remedy = legal_remedy
        self.related_department = related_department
        self.annotations = annotations
    
    @classmethod
    def from_json(cls, annotated_decision_path: str) -> 'AnnotatedDecision':
        annotated_decision_dict = read_json_file(annotated_decision_path)
 
        annotated_sections = [
            AnnotedSection(section['class_id'], section['text']) 
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
 
            # print(annotated_section.class_id)
            # print(annotated_section.text)
            # print('\n\n\n\n\n\n\n\n\n\n')
 
        return document
 
 
def main():
    load_dotenv()
 
    OPENAI_KEY=getenv("OPENAI_KEY")
    print(OPENAI_KEY)
 
    annotated_decision = AnnotatedDecision.from_json(join('documents', 'annotated_decisions', 'ste_1537-2023.json'))
 
    print(annotated_decision.document_id)
 
    
 
main()