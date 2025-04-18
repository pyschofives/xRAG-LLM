import re
import json

def clean_text(text):
    """
    Clean the raw text: remove extra spaces, unwanted characters, etc.
    """
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces
    text = text.strip()               # Remove leading/trailing spaces
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
    return text

def process_samples(raw_samples, max_length=512):
    """
    Process raw samples into clean passages for embedding.
    
    Arguments:
    - raw_samples: Raw list of {question, context, answer}
    - max_length: Max token length for each passage.
    
    Returns:
    - A list of processed samples, ready for embedding
    """
    processed = []
    for sample in raw_samples:
        question = clean_text(sample.get("question", ""))
        context = clean_text(sample.get("context", ""))
        answer = clean_text(sample.get("answer", ""))
        
        if question and context:
            # Combine question and context for retrieval
            combined = f"Question: {question} Context: {context}"
            
            # Truncate if the combined context is too long
            if len(combined.split()) > max_length:
                combined = " ".join(combined.split()[:max_length])
            
            processed.append({
                "question": question,
                "context": context,
                "answer": answer,
                "combined": combined
            })
    
    return processed

# Example: Process the loaded TriviaQA and NaturalQ data
def process_all_data(trivia_samples, nq_samples):
    print("📂 Processing TriviaQA samples...")
    processed_triviaqa = process_samples(trivia_samples)
    
    print("📂 Processing NaturalQ samples...")
    processed_naturalq = process_samples(nq_samples)

    return processed_triviaqa + processed_naturalq  # Combine both datasets

if __name__ == "__main__":
    # Assuming trivia_samples and nq_samples are already loaded
    print("📂 Loading TriviaQA and NaturalQ datasets...")

    # Use the raw samples directly if already loaded
    from rag_pipeline.data_loader import load_triviaqa, load_naturalq
    
    trivia_samples = load_triviaqa()
    nq_samples = load_naturalq()

    # Process and combine the data
    processed_samples = process_all_data(trivia_samples, nq_samples)

    print(f"✅ Processed {len(processed_samples)} samples.")

    # Optionally, save the processed data
    with open('processed_data.json', 'w') as f:
        json.dump(processed_samples, f)

    print("✅ Processed data saved as 'processed_data.json'.")
