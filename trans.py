from deep_translator import GoogleTranslator
from LLMmain import get_docs, generate_answer

def sinhalaToEnglish(query: str) -> str:
    """Translate Sinhala to English."""
    translator = GoogleTranslator(source="si", target="en")
    translated_query = translator.translate(query)
    return translated_query

def englishToSinhala(text: str) -> str:
    """Translate English to Sinhala."""
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)