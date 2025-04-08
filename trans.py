from deep_translator import GoogleTranslator

# Translate Sinhala to English
def sinhalaToEnglish(query: str) -> str:
    translator = GoogleTranslator(source="si", target="en")
    translated_query = translator.translate(query)
    return translated_query

# Translate English to Sinhala
def englishToSinhala(text: str) -> str:
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)