import os
import tensorflow_hub as hub
import numpy as np

def evaluate_with_use(reference, candidate):
    print("Loading Universal Sentence Encoder model...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Loading finished")
    embeddings = model([reference, candidate])
    similarity = np.inner(embeddings[0], embeddings[1])
    return similarity

def measure_length(summary):
    """Measure the word count of a summary."""
    return len(summary.split())

def read_summary_from_file(file_path: str) -> str:
    """Read a summary from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reference_folder = os.path.join(base_dir, "documents", "ste_summaries")
    candidate_folder_plain = os.path.join(base_dir, "documents", "plain_summaries")
    candidate_folder_my = os.path.join(base_dir, "documents", "my_summaries")

    decision_number = "ste_1414-2024"
    reference_file_path = os.path.join(reference_folder, f"{decision_number}.txt")

    candidate_file_path_plain = os.path.join(candidate_folder_plain, f"{decision_number}.txt")
    candidate_file_path_my = os.path.join(candidate_folder_my, f"{decision_number}.txt")

    reference_summary = read_summary_from_file(reference_file_path)
    candidate_summary_plain = read_summary_from_file(candidate_file_path_plain)
    candidate_summary_my = read_summary_from_file(candidate_file_path_my)

    # Evaluate MoverScore
    score_plain = evaluate_with_use(reference_summary, candidate_summary_plain)
    score_my = evaluate_with_use(reference_summary, candidate_summary_my)

    print(f"USE Semantic Similarity Plain: {score_plain:.4f}")
    print(f"USE Semantic Similarity My: {score_my:.4f}")

    # Print length
    plain_length = measure_length(candidate_summary_plain)
    my_length = measure_length(candidate_summary_my)
    print(f"PLAIN Summary Length: {plain_length} words")
    print(f"MY Summary Length: {my_length} words")

if __name__ == "__main__":
    main()
