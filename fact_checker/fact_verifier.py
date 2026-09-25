from duckduckgo_search import DDGS
import ollama

def verify_claim(claim):
    if not claim or "NO_CLAIM" in claim:
        return None

    # Step 1: Fast DuckDuckGo Search
    evidence = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(claim, max_results=2))
            evidence = "\n".join([r['body'] for r in results])
    except Exception as e:
        evidence = "Search unavailable."

    # Step 2: Ollama Fact Checking
    prompt = f"""
    Fact Claim: {claim}
    Search Evidence: {evidence}

    Is the claim correct based on facts?
    Reply strictly in this format:
    VERDICT: [CORRECT / INCORRECT]
    FACT: [1 short sentence explaining the true fact]
    """

    try:
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"VERDICT: UNVERIFIED\nFACT: Could not connect to local model. {e}"