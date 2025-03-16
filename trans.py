from deep_translator import GoogleTranslator
import re

# Medical and technical terms to keep in English
MEDICAL_TERMS = [
    "postpartum depression", "PPD","(PPD)" , "postpartum", "anxiety", "panic attacks",
    "fatigue", "appetite", "post-traumatic stress", "PTSD", "oxytocin",
    "serotonin", "dopamine", "hormone", "thyroid", "estrogen", "progesterone",
    "antidepressant", "SSRI", "psychiatrist", "psychologist", "therapy",
    "cognitive behavioral therapy", "CBT", "prenatal", "postnatal", "symptoms",
    "diagnosis", "treatment", "medication", "breastfeeding", "lactation"
]

# Dictionary for term replacements in Sinhala translation
SINHALA_TERM_REPLACEMENTS = {
    "අවපාතය": "මානසික අවපීඩනය",  # Better term for depression
    "රෝග ලක්ෂණ": "ලක්ෂණ",  # Simpler term for symptoms
    "පසු ප්‍රසව": "ප්‍රසූතිය පසු",  # Corrected term for postpartum
}

def sinhalaToEnglish(query: str) -> str:
    """
    Translate Sinhala to English while preserving medical terms.
    
    Args:
        query: Text in Sinhala to translate
        
    Returns:
        Translated text in English with preserved medical terms
    """
    try:
        translator = GoogleTranslator(source='si', target='en')
        translated_query = translator.translate(query)
        
        # Ensure medical terms are properly capitalized and preserved
        for term in MEDICAL_TERMS:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            translated_query = pattern.sub(term, translated_query)
            
        return translated_query
    
    except Exception as e:
        print(f"Translation error (SI to EN): {e}")
        # Return original text if translation fails
        return query

def englishToSinhala(text: str) -> str:
    """
    Translate English to Sinhala while preserving the first sentence in English
    and preserving medical terms and improving terminology.
    
    Args:
        text: Text in English to translate
        
    Returns:
        Text with first sentence in English and the rest translated to Sinhala
    """
    try:
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return text
        
        # Keep the first sentence as is
        first_sentence = sentences[0]
        
        # If there are more sentences, translate them
        if len(sentences) > 1:
            # Join the remaining sentences
            rest_of_text = " ".join(sentences[1:])
            
            # Use a simpler approach - directly translate, then replace medical terms
            translator = GoogleTranslator(source='en', target='si')
            translated_rest = translator.translate(rest_of_text)
            
            # Find and replace medical terms with proper formatting
            for term in MEDICAL_TERMS:
                # First translate the term to see how it appears in Sinhala
                try:
                    translated_term = translator.translate(term)
                    # Replace the translated term with the original English term
                    if translated_term and translated_term in translated_rest:
                        translated_rest = translated_rest.replace(translated_term, term)
                except:
                    continue
            
            # Apply Sinhala terminology improvements
            for incorrect, improved in SINHALA_TERM_REPLACEMENTS.items():
                translated_rest = translated_rest.replace(incorrect, improved)
            
            # Combine first sentence with translated rest
            return first_sentence + " " + translated_rest
        else:
            # If there's only one sentence, return it unchanged
            return first_sentence
    
    except Exception as e:
        print(f"Translation error (EN to SI): {e}")
        # Return original text if translation fails
        return text