import ollama

def extract_claim(text):
    if not text or len(text.strip()) < 3:
        return "NO_CLAIM"
    
    text_clean = text.strip()
    text_lower = text_clean.lower()

    # Fast check for simple non-factual utterances
    if text_lower in ["thank you.", "thank you", "hello", "hi", "yes", "no", "okay", "thanks"]:
        return "NO_CLAIM"

    # Strict strict prompt remove karke simple extractor lagayein
    prompt = f"""Is this sentence making a factual statement that can be true or false?
Sentence: '{text_clean}'

If YES, output the sentence as it is.
If NO (e.g. greeting, question, or opinion), output strictly 'NO_CLAIM'."""

    try:
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        result = response['message']['content'].strip()
        
        # Fallback: Agar LLM confuse ho jaye toh direct text paas kar dein
        if "NO_CLAIM" not in result:
            return text_clean
        return "NO_CLAIM"
    except Exception as e:
        print(f"Extraction Error: {e}")
        return text_clean