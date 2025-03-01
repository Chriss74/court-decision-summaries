import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_KEY = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=OPENAI_KEY)

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate the number of tokens in the text."""
    from tiktoken import encoding_for_model
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))

def split_text_into_chunks(text: str, max_tokens: int) -> list:
    """Split text into smaller chunks based on token limits."""
    sentences = text.split(". ")
    chunks, chunk, chunk_token_size = [], "", 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if chunk_token_size + sentence_tokens > max_tokens:
            chunks.append(chunk.strip())
            chunk, chunk_token_size = sentence + ". ", sentence_tokens
        else:
            chunk += sentence + ". "
            chunk_token_size += sentence_tokens

    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize_chunk(chunk: str, client, prompt: str, temperature: float, max_tokens: int) -> str:
    """Summarize a single chunk using OpenAI."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def summarize_large_text(decision_number: str, input_folder: str, output_folder: str, prompt: str, temperature=0.7, max_chunk_tokens=2500, summary_tokens=700, max_total_tokens=13000):
    # file paths
    input_path = os.path.join(input_folder, f"{decision_number}.txt")
    output_path = os.path.join(output_folder, f"{decision_number}.txt")

    # Ensure file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")

    # Create folder if not exist
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text
    chunks = split_text_into_chunks(text, max_chunk_tokens)

    # Summarize and combine
    combined_summary = ""
    total_tokens = 0
    for idx, chunk in enumerate(chunks):
        print(f"Summarizing chunk {idx + 1} of {len(chunks)}...")

        # Estimate the tokens limit
        remaining_tokens = max_total_tokens - total_tokens
        if remaining_tokens <= 0:
            print("Token limit reached. Stopping summarization.")
            break

        # Limit the summary tokens to the remaining allowed tokens
        current_summary_tokens = min(summary_tokens, remaining_tokens)
        summary = summarize_chunk(chunk, client, prompt, temperature, current_summary_tokens)
        combined_summary += summary + "\n\n"

        # Update the total token count
        total_tokens += estimate_tokens(summary)

    # Save the combined summary to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(combined_summary)

    print(f"Summary saved to {output_path}. Total tokens used: {total_tokens}")
    return combined_summary

def main():
    decision_number = "ste_2325-2023"
    input_folder = "documents/txt_files"
    output_folder = "documents/plain_summaries"
    prompt = "You are a legal professional. Summarize the following legal text in a concise manner and in Greek language, keeping all key legal concepts."

    try:
        # Summarize the large text file
        summarize_large_text(decision_number, input_folder, output_folder, prompt)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
