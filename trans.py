from deep_translator import GoogleTranslator

def sinhalaToEnglish(query: str) -> str:
    translator = GoogleTranslator(source="si", target="en")
    translated_query = translator.translate(query)
    return translated_query

def englishToSinhala(text: str) -> str:
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)
