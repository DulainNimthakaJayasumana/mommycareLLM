"""
evaluation_metrics.py

This script evaluates generated text against a reference text using four metrics:
  • ROUGE: Measures n‑gram overlaps (recall) between generated and reference text.
  • METEOR: Aligns words using exact matches, synonyms, and stems.
  • BERTScore: Uses contextual embeddings to capture semantic similarity.
  • CIDEr: Uses TF-IDF weighted n‑gram matching (commonly used in caption evaluation).

Before running this script, install the required packages:
    pip install rouge-score nltk bert-score

For CIDEr, you need the pycocoevalcap package. Clone and install it as follows:
    git clone https://github.com/salaniz/pycocoevalcap.git
    cd pycocoevalcap
    pip install -e .

If CIDEr is not installed, the code will skip its evaluation.
"""

import ssl
import nltk
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
from bert_score import score as bert_score
from pycocoevalcap.cider.cider import Cider

# Fix SSL certificate verification issue for NLTK downloads (if needed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data resources, including 'punkt_tab'
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# Try importing CIDEr from pycocoevalcap
try:
    from pycocoevalcap.cider.cider import Cider
    cider_available = True
except ImportError:
    cider_available = False
    print("CIDEr evaluation not available. Please install pycocoevalcap for CIDEr scores.")

def evaluate_rouge(generated: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

def evaluate_meteor(generated: str, reference: str) -> float:
    """
    Compute the METEOR score.
    Tokenize both the generated and reference texts before computing the score.
    """
    # Tokenize using nltk.word_tokenize
    tokenized_generated = nltk.word_tokenize(generated)
    tokenized_reference = nltk.word_tokenize(reference)
    # Pass the token lists to the meteor function.
    score = meteor_score.single_meteor_score(tokenized_reference, tokenized_generated)
    return score

def evaluate_bert_score(generated: str, reference: str) -> float:
    """
    Compute the BERTScore F1.
    """
    # bert_score.score() expects lists of strings.
    P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
    return F1.item()

def evaluate_cider(generated: str, reference: str):
    """
    Compute the CIDEr score.
    The pycocoevalcap package expects inputs as dictionaries.
    """
    if not cider_available:
        print("CIDEr metric is not available. Skipping CIDEr evaluation.")
        return None
    # Prepare dictionaries with one key-value pair.
    sample_id = "0"
    gts = {sample_id: [reference]}  # list of reference sentences
    res = {sample_id: [generated]}  # list of generated sentences
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score

if __name__ == '__main__':
    # Example texts (modify these with your own generated and reference texts)
    generated_text = (
        "Secondary postpartum hemorrhage occurs when heavy bleeding happens after 24 hours up to 12 weeks postpartum."
    )
    reference_text = (
        "Excessive bleeding from the birth canal occurring between 24 hours and 12 weeks postnatally."
    )

    # Evaluate each metric
    rouge_scores = evaluate_rouge(generated_text, reference_text)
    meteor = evaluate_meteor(generated_text, reference_text)
    bert = evaluate_bert_score(generated_text, reference_text)
    cider = evaluate_cider(generated_text, reference_text)

    # Print the evaluation results
    print("ROUGE scores:")
    for key, value in rouge_scores.items():
        print(f"  {key}: {value}")

    print("\nMETEOR score:", meteor)
    print("BERTScore (F1):", bert)
    print("CIDEr score:", cider)