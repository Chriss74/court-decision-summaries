from sentence_transformers import SentenceTransformer, util
import os 

def evaluate_with_sbert(reference_summary, candidate_summary, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    reference_embedding = model.encode(reference_summary, convert_to_tensor=True)
    candidate_embedding = model.encode(candidate_summary, convert_to_tensor=True)
    similarity_score = util.cos_sim(reference_embedding, candidate_embedding).item()
    return similarity_score

def measure_length(summary):
    return len(summary.split())

def read_summary_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reference_folder = os.path.join(base_dir, "documents", "ste_summaries")
    candidate_folder_plain = os.path.join(base_dir, "documents", "plain_summaries")
    candidate_folder_my = os.path.join(base_dir, "documents", "my_summaries")
    results_file_path = os.path.join(base_dir, "results.txt")

    decision_numbers = ["ste_1412-2024", "ste_1508-2024", "ste_1537-2023", "ste_2221-2023", "ste_2325-2023"]

    total_similarity_plain = 0
    total_similarity_my = 0
    count = 0

    with open(results_file_path, "w", encoding="utf-8") as results_file:
        for decision_number in decision_numbers:
            print(f"Processing decision: {decision_number}")
            
            reference_file_path = os.path.join(reference_folder, f"{decision_number}.txt")
            candidate_file_path_plain = os.path.join(candidate_folder_plain, f"{decision_number}.txt")
            candidate_file_path_my = os.path.join(candidate_folder_my, f"{decision_number}.txt")

            try:
                reference_summary = read_summary_from_file(reference_file_path)
                candidate_summary_plain = read_summary_from_file(candidate_file_path_plain)
                candidate_summary_my = read_summary_from_file(candidate_file_path_my)
            
                # Evaluate similarity 
                similarity_score_plain = evaluate_with_sbert(reference_summary, candidate_summary_plain)
                similarity_score_my = evaluate_with_sbert(reference_summary, candidate_summary_my)
                
                total_similarity_plain += similarity_score_plain
                total_similarity_my += similarity_score_my
                count += 1
                
                results_file.write(f"{decision_number}: PLAIN {similarity_score_plain:.4f}, MY {similarity_score_my:.4f}\n")
                
                print(f"Semantic Similarity Score (SBERT) PLAIN: {similarity_score_plain:.4f}")
                print(f"Semantic Similarity Score (SBERT) MY: {similarity_score_my:.4f}")
                
                # Print length
                plain_length = measure_length(candidate_summary_plain)
                my_length = measure_length(candidate_summary_my)
                print(f"PLAIN Summary Length: {plain_length} words")
                print(f"MY Summary Length: {my_length} words")
                print("-")
            
            except FileNotFoundError as e:
                print(e)

        if count > 0:
            avg_similarity_plain = total_similarity_plain / count
            avg_similarity_my = total_similarity_my / count
            results_file.write(f"\nCollective Scores - PLAIN: {avg_similarity_plain:.4f}, MY: {avg_similarity_my:.4f}\n")
            print(f"Collective Scores - PLAIN: {avg_similarity_plain:.4f}, MY: {avg_similarity_my:.4f}")

if __name__ == "__main__":
    main()
