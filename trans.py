from deep_translator import GoogleTranslator

#Translate Sinhala to English.
def sinhalaToEnglish(query: str) -> str:

    translator = GoogleTranslator(source="si", target="en")
    translated_query = translator.translate(query)
    return translated_query

#Translate English to Sinhala.
def englishToSinhala(text: str) -> str:
   
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)


# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# class BilingualTranslator:
#     def __init__(self):
#         # Use the smaller 600M parameter version of NLLB
#         self.model_name = "facebook/nllb-200-distilled-600M"
#         print(f"Loading model: {self.model_name}...")
        
#         # Load model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
#         # Language codes
#         self.sinhala_code = "sin_Sinh"  # Sinhala
#         self.english_code = "eng_Latn"  # English
        
#         print("Translation model loaded successfully")
    
#     def sinhala_to_english(self, text):
#         """Translate from Sinhala to English"""
#         try:
#             # Skip translation if text is empty
#             if not text.strip():
#                 return text
                
#             return self._translate(text, src_lang=self.sinhala_code, tgt_lang=self.english_code)
#         except Exception as e:
#             print(f"Error in Sinhala to English translation: {str(e)}")
#             return text  # Return original text on error
    
#     def english_to_sinhala(self, text):
#         """Translate from English to Sinhala"""
#         try:
#             # Skip translation if text is empty
#             if not text.strip():
#                 return text
                
#             return self._translate(text, src_lang=self.english_code, tgt_lang=self.sinhala_code)
#         except Exception as e:
#             print(f"Error in English to Sinhala translation: {str(e)}")
#             return text  # Return original text on error
    
#     def _translate(self, text, src_lang, tgt_lang):
#         """Internal method that handles translation between languages"""
#         # Handle batch inputs (if text is a list)
#         if isinstance(text, list):
#             return [self._translate(t, src_lang, tgt_lang) for t in text]
        
#         # Set the source language for tokenizer
#         self.tokenizer.src_lang = src_lang
        
#         # Encode input text
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512
#         )
        
#         # Generate translation with target language
#         with torch.no_grad():
#             # Get the token ID for the target language
#             forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
            
#             outputs = self.model.generate(
#                 **inputs,
#                 forced_bos_token_id=forced_bos_token_id,
#                 max_length=512,
#                 num_beams=5,  # Use beam search for better quality
#                 length_penalty=1.0,
#                 early_stopping=True
#             )
        
#         # Decode translated text
#         translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         return translation

# # Initialize the translator with error handling
# try:
#     translator = BilingualTranslator()
# except Exception as e:
#     print(f"Failed to initialize translator: {str(e)}")
#     # Create fallback functions that return the original text
#     def sinhalaToEnglish(query: str) -> str:
#         print("WARNING: Translation service unavailable. Using original text.")
#         return query
    
#     def englishToSinhala(text: str) -> str:
#         print("WARNING: Translation service unavailable. Using original text.")
#         return text
# else:
#     # Define the wrapper functions with additional error handling
#     def sinhalaToEnglish(query: str) -> str:
#         """Translate Sinhala to English with fallback."""
#         try:
#             # Basic language detection (simple heuristic)
#             has_sinhala_chars = any(0x0D80 <= ord(char) <= 0x0DFF for char in query)
            
#             # If no Sinhala characters and text is not empty, assume it's already English
#             if not has_sinhala_chars and query.strip():
#                 print("Text appears to be non-Sinhala. Returning original.")
#                 return query
                
#             return translator.sinhala_to_english(query)
#         except Exception as e:
#             print(f"Translation error: {str(e)}")
#             return query
    
#     def englishToSinhala(text: str) -> str:
#         """Translate English to Sinhala with fallback."""
#         try:
#             # Skip empty or very short text
#             if len(text.strip()) < 2:
#                 return text
                
#             return translator.english_to_sinhala(text)
#         except Exception as e:
#             print(f"Translation error: {str(e)}")
#             return text