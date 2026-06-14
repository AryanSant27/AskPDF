import time

def generate_content_with_retry(gemini_model, prompt, max_retries=3, initial_delay=2.0):
    """
    Wrapper around gemini_model.generate_content to handle 429 Rate Limits / Quota Exhaustion
    using exponential backoff.
    """
    if not gemini_model:
        raise ValueError("Gemini model is not configured.")
        
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return gemini_model.generate_content(prompt)
        except Exception as e:
            err_str = str(e).lower()
            # Catch 429, resource_exhausted, quota, or rate limit exceeded strings
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "limit" in err_str:
                if attempt < max_retries - 1:
                    print(f"[Gemini Retry] Rate limit hit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2.0
                    continue
            raise e
