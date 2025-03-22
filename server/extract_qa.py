import json

# File paths
INPUT_FILE = "test_run_embeddings.jsonl"  # Your raw dataset
OUTPUT_FILE = "formatted_qa.jsonl"  # Output file for cleaned Q&A pairs

def extract_and_format_qa():
    """Extracts and formats Q&A pairs for embeddings."""
    formatted_texts = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)  # Load JSONL line as dictionary
            messages = item.get("messages", [])

            question = None
            answer = None

            # Extract user question and assistant answer
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content")
                elif msg.get("role") == "assistant":
                    answer = msg.get("content")

            # Skip if either question or answer is missing
            if not question or not answer:
                continue  

            # Format the Q&A text
            formatted_text = f"Question: {question}\nAnswer: {answer}"
            formatted_texts.append({"id": str(len(formatted_texts)), "text": formatted_text})

    # Save formatted Q&A pairs
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for record in formatted_texts:
            out_f.write(json.dumps(record) + "\n")

    print(f"✅ Extracted {len(formatted_texts)} Q&A pairs and saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    extract_and_format_qa()
