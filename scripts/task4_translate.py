import os
from dotenv import load_dotenv
from sarvamai import SarvamAI


# Environment setup

# Load variables from .env file
load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if SARVAM_API_KEY is None:
    raise EnvironmentError(
        "SARVAM_API_KEY not found. Please set it in the .env file."
    )


# Translation function

def translate_to_english(text: str, source_language_code: str = "auto") -> str:

    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    # Initialize Sarvam client
    client = SarvamAI(
        api_subscription_key=SARVAM_API_KEY
    )

    # Call Sarvam Translation API
    response = client.text.translate(
        input=text,
        source_language_code=source_language_code,
        target_language_code="en-IN"  
    )

    # Sarvam SDK returns a TranslationResponse object
    return response.translated_text


# Local test 

if __name__ == "__main__":
    sample_text = "मुझे कृत्रिम बुद्धिमत्ता के बारे में बताओ"
    translated_text = translate_to_english(sample_text)
    print("Translated Text:", translated_text)






