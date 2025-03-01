from rouge_score import rouge_scorer
import os

def read_summary_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def compute_rouge_scores(reference_summary, candidate_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference_summary, candidate_summary)

def main():
    # Base directory for the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the directories for the summaries
    reference_folder = os.path.join(base_dir, "documents", "ste_summaries")
    candidate_folder_plain = os.path.join(base_dir, "documents", "plain_summaries")
    candidate_folder_my = os.path.join(base_dir, "documents", "my_summaries")
    output_file = os.path.join(base_dir, "documents", "rouge_scores.txt")

    # List of decision numbers to evaluate
    decision_numbers = ["ste_1412-2024", "ste_1508-2024", "ste_1537-2023", "ste_2221-2023", "ste_2325-2023"]
    total_scores_plain = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    total_scores_my = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with open(output_file, "w", encoding="utf-8") as out_file:
        for decision_number in decision_numbers:
            try:
                # Construct the full file paths
                reference_file_path = os.path.join(reference_folder, f"{decision_number}.txt")
                candidate_file_path_plain = os.path.join(candidate_folder_plain, f"{decision_number}.txt")
                candidate_file_path_my = os.path.join(candidate_folder_my, f"{decision_number}.txt")

                reference_summary = read_summary_from_file(reference_file_path)
                candidate_summary_plain = read_summary_from_file(candidate_file_path_plain)
                candidate_summary_my = read_summary_from_file(candidate_file_path_my)

                # Compute ROUGE scores for plain summary
                scores_plain = compute_rouge_scores(reference_summary, candidate_summary_plain)
                scores_my = compute_rouge_scores(reference_summary, candidate_summary_my)

                # Store scores for averaging
                for key in total_scores_plain:
                    total_scores_plain[key].append(scores_plain[key].fmeasure)
                    total_scores_my[key].append(scores_my[key].fmeasure)

                # Write results to file
                out_file.write(f"ROUGE Scores for {decision_number} (Plain):\n")
                for key, value in scores_plain.items():
                    out_file.write(f"{key}: Precision: {value.precision:.2f}, Recall: {value.recall:.2f}, F1: {value.fmeasure:.2f}\n")
                out_file.write("\n")

                out_file.write(f"ROUGE Scores for {decision_number} (My):\n")
                for key, value in scores_my.items():
                    out_file.write(f"{key}: Precision: {value.precision:.2f}, Recall: {value.recall:.2f}, F1: {value.fmeasure:.2f}\n")
                out_file.write("\n-----------------------------\n\n")

                print(f"ROUGE scores for {decision_number} saved successfully.")
            except FileNotFoundError as e:
                print(f"Error: {e}")

        # Compute and write average scores
        out_file.write("Average ROUGE Scores:\n")
        for key in total_scores_plain:
            avg_plain = sum(total_scores_plain[key]) / len(total_scores_plain[key]) if total_scores_plain[key] else 0
            avg_my = sum(total_scores_my[key]) / len(total_scores_my[key]) if total_scores_my[key] else 0
            out_file.write(f"{key} (Plain): F1: {avg_plain:.2f}\n")
            out_file.write(f"{key} (My): F1: {avg_my:.2f}\n")
        out_file.write("\n-----------------------------\n\n")

if __name__ == "__main__":
    main()
