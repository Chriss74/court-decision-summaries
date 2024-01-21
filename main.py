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
    with open(file_path) as json_file:
        return json.load(json_file)
 
class TokenEstimator:
    def __init__(self, model: str = 'gpt-3.5-turbo') -> None:
        self.encoding = tiktoken.encoding_for_model(model)
 
    def estimate(self, text: str) -> int:
        return len(self.encoding.encode(text))
 
class AnnotedSection:
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
            annotations: List[AnnotedSection]) -> None:
        
        self.document_id = document_id
        self.court = court
        self.legal_remedy = legal_remedy
        self.related_department = related_department
        self.annotations = annotations
    
    @classmethod
    def from_json(cls, annotated_decision_path: str, annotated_mappings_dict: Dict[str, Any]) -> 'AnnotatedDecision':
        annotated_decision_dict = read_json_file(annotated_decision_path)
 
        annotated_sections = [
            AnnotedSection(
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
            # print(annotated_section.class_id)
            # print(annotated_section.text)
            # print('\n\n\n\n\n\n\n\n\n\n')
 
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
 
def openaicompletion(client: OpenAI, text):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful legal professional'
            },
            {
                'role': 'user',
                #'content': f'Summarize this:\n{text}'
                'content': f'Summarize and translate to Greek:\n{text}'
            }
        ],
    )
 
    return response
 
 
def main():
    load_dotenv()
 
    OPENAI_KEY=getenv('OPENAI_KEY')
    print(OPENAI_KEY)
 
    OPENAI_TOKEN_LIMIT = 4050
 
    annotation_mappings = read_json_file(join('documents', 'annotation_mappings.json'))
 
    annotated_decision = AnnotatedDecision.from_json(join('documents', 'annotated_decisions', 'ste_1537-2023.json'), annotation_mappings)
 
    print(annotated_decision.document_id)
 
    client = OpenAI(api_key=OPENAI_KEY)
 
    text = "11. Επειδή, με την 686/2018 απόφαση της Ολομέλειας του Συμβουλίου της Επικρατείας κρίθηκε ότι από τον συνδυασμό των διατάξεων των άρθρων 4 παρ. 1, 5 παρ. 1 και 16 του Συντάγματος δεν αποκλείεται η πρόβλεψη από τον νομοθέτη ευνοϊκών ρυθμίσεων για τους αθλητές που έχουν επιτύχει εξαιρετικές αγωνιστικές διακρίσεις σε σημαντικές αθλητικές διοργανώσεις, όπως η θέσπιση ενός συστήματος πριμοδότησης βαθμολογίας για την εισαγωγή τους στην τριτοβάθμια εκπαίδευση, και μάλιστα, όχι μόνο σε συναφείς με τον αθλητισμό σχολές (π.χ. Τ.Ε.Φ.Α.Α.), αλλά και στα λοιπά Α.Ε.Ι. και Τ.Ε.Ι. της χώρας, προκειμένου οι υποψήφιοι αυτοί, λόγω της απαιτούμενης για την επίτευξη των ως άνω διακρίσεων έντονης ενασχολήσεώς τους με τον αθλητισμό, να μην βρεθούν σε ήσσονα μοίρα έναντι των λοιπών υποψηφίων, σχετικά με την επιλογή των σπουδών τους και τις βάσει αυτών περαιτέρω επαγγελματικές τους επιδιώξεις. Περαιτέρω έγινε δεκτό ότι ο καθορισμός των κατηγοριών διακεκριμένων αθλητών, στους οποίους παρέχεται το σχετικό ευεργέτημα ανήκει στην ευχέρεια του νομοθέτη και υπόκειται σε έλεγχο ορίων από τα δικαστήρια, όπως και οι θεσπιζόμενες σχετικές ευνοϊκές ρυθμίσεις, οι οποίες πρέπει, πάντως, να τελούν σε αρμονία προς την υποχρέωση του νομοθέτη να εξασφαλίζει την πρόσβαση στην τριτοβάθμια εκπαίδευση προσώπων κεκτημένων τα αναγκαία εφόδια για την ενεργό παρακολούθηση της θεωρητικής και πρακτικής διδασκαλίας των ανωτάτων εκπαιδευτικών ιδρυμάτων, ώστε να επιτυγχάνεται η αποστολή των ιδρυμάτων αυτών και η εύρυθμη λειτουργία τους. Ενόψει των ανωτέρω, με την ως άνω 686/2018 απόφαση της Ολομελείας κρίθηκε ότι η διάκριση που θεσπίζει η διάταξη της παρ. 8 του άρθρου 34 του ν. 2725/1999 υπέρ των αθλητών υποψηφίων που έχουν επιτύχει τις προβλεπόμενες από τον νόμο διακρίσεις (καθιέρωση συστήματος προσαύξησης της βαθμολογίας τους ποσοστιαίως, ανάλογα με την επιτευχθείσα αγωνιστική διάκριση), αναφερόμενη σε κατηγορίες υποψηφίων που προσδιορίζονται με αντικειμενικά κριτήρια, δικαιολογείται από λόγους δημοσίου συμφέροντος και, συγκεκριμένα, από την ανάγκη παροχής κινήτρων στους ασχολούμενους με τον αθλητισμό, για την ανάπτυξη του οποίου οφείλει να μεριμνά η Πολιτεία, κατ’ άρθρο 16 παρ. 9 του Συντάγματος και είναι, κατ’ αρχήν, συνταγματικώς θεμιτή, εφόσον, μάλιστα, με τη διάταξη αυτή δεν θίγεται ο αριθμός των κανονικώς εισακτέων στα τμήματα και τις σχολές των Α.Ε.Ι. και Τ.Ε.Ι., καθόσον το καθοριζόμενο από αυτήν ποσοστό υπέρ της ως άνω κατηγορίας υποψηφίων τίθεται καθ’ υπέρβαση του ολικού αριθμού εισακτέων για κάθε σχολή ή τμήμα σχολής των Α.Ε.Ι. και Τ.Ε.Ι. Περαιτέρω, όμως, κατά τα γενόμενα δεκτά με την ίδια ως άνω 686/2018 απόφαση του Συμβουλίου της Επικρατείας, οι εν λόγω διατάξεις του άρθρου 34 του ν. 2725/1999, όπως αυτό ίσχυε μετά την τροποποίησή του με το άρθρο 17 παρ. 1 του ν. 4429/2016, αντίκεινται στις συνταγματικές αρχές της ισότητας, της αξιοκρατίας και της ορθολογικής οργανώσεως της παρεχόμενης εκπαιδεύσεως, σε συνδυασμό με τη συνταγματική αρχή της αναλογικότητας μέτρου/σκοπού, κατά το μέρος που προβλέπουν: α) την πριμοδότηση αθλητών ακόμα και για επιτυχίες σε αθλητικούς αγώνες ήσσονος σημασίας, β) την καθ’ υπέρβαση εισαγωγή αθλητών στις ανώτατες σχολές στο δυσανάλογα υψηλό ποσοστό του 4,5% των προβλεφθεισών θέσεων, το οποίο παρίσταται αυθαίρετο ενόψει και του πάγιου χαρακτήρα της ρυθμίσεως αυτής, η οποία δεν προϋποθέτει οποιαδήποτε επίκαιρη εκτίμηση της δυνατότητας μιας σχολής για απορρόφηση σπουδαστών και γ) την εισαγωγή διακριθέντος αθλητή σε σχολή της προτιμήσεώς του, εφόσον συγκεντρώσει αριθμό μορίων τουλάχιστον ίσο με το 90% του αριθμού των μορίων του τελευταίου εισαχθέντος στη συγκεκριμένη σχολή κατά το ίδιο ακαδημαϊκό έτος, μετά την προσαύξηση του συνόλου των μορίων του από τις αθλητικές του διακρίσεις. Τούτο δε, διότι - όπως κρίθηκε - οι ανωτέρω ρυθμίσεις υπερακοντίζουν τον δημοσίου συμφέροντος σκοπό που υπηρετεί η διάταξη (προαγωγή του αθλητισμού), λόγω αφενός του επιβαλλόμενου με αυτές σημαντικού περιορισμού προσβάσεως στην τριτοβάθμια εκπαίδευση των λοιπών υποψηφίων φοιτητών που δεν είναι αθλητές και, αφετέρου, της σημαντικής υποχωρήσεως των ακαδημαϊκών κριτηρίων πρόσβασης, την οποία συνεπάγονται. Συνεπώς, το Δικαστήριο έκρινε με την απόφαση αυτή (686/2018) ότι η επίμαχη διάταξη της παρ. 8 του άρθρου 34 του ν. 2725/1999 παραβιάζει τα άρθρα 4 παρ. 1, 5 παρ. 1 και 16 του Συντάγματος, κατά το μέρος που οι εν λόγω ρυθμίσεις υπερβαίνουν, κατά τα ανωτέρω, τα ακραία όρια πέραν των οποίων το παρεχόμενο από τη διάταξη αυτή προνόμιο εμφανίζεται δικαιολογημένο και επιτρεπτό συνταγματικώς."
 
    
    token_estimator = TokenEstimator()
 
    decision_tokens = annotated_decision.get_document_tokens(token_estimator)
    print(decision_tokens)
 
    summary_tokens = min(decision_tokens * 0.05, 11000)
    print(summary_tokens)
 
    all_importance_times_tokens = sum([(annotation.importance * token_estimator.estimate(annotation.text)) for annotation in annotated_decision.annotations])
    print(all_importance_times_tokens)
 
    for section in annotated_decision.annotations:
        print(section.class_name)
        section_input_tokens = token_estimator.estimate(section.text)
        importance_times_tokens = section_input_tokens * section.importance
        portion = importance_times_tokens / all_importance_times_tokens
        
        section_summary_tokens = portion*summary_tokens
        
        if section_summary_tokens + section_input_tokens > OPENAI_TOKEN_LIMIT:
            sentences = split_to_sentences(section.text)
            
            n_chunks = math.ceil((section_summary_tokens + section_input_tokens) / OPENAI_TOKEN_LIMIT)
 
            max_chunk_token_size = (section_summary_tokens/ n_chunks) + (section_input_tokens / n_chunks)
 
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
 
 
            merged_summary = ''
            for item in chunks:
                response = openaicompletion(client, item)
                summary = response.choices[0].message.content
 
                merged_summary = f'{merged_summary} {summary}'
            print(merged_summary)
            
            merged_summaries = openaicompletion(client, merged_summary)
            summary_of_summaries = merged_summaries.choices[0].message.content
            print(summary_of_summaries)
 
            
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