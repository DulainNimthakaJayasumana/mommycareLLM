import csv
import os
from LLMmain import get_docs, generate_answer
from trans import sinhalaToEnglish, englishToSinhala
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu

# Load environment variables from .env
load_dotenv()

def load_test_cases(csv_path: str) -> list:
    """
    Load test cases from a CSV file.
    CSV should have columns: 'question' and 'ground_truth'
    """
    test_cases = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_cases.append(row)
    return test_cases

def evaluate_bleu_score(reference: str, candidate: str) -> float:
    """
    Compute the BLEU score for a candidate answer against the reference.
    (Tokenization here is done via simple split; consider using a better tokenizer for production.)
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    # BLEU score expects a list of reference token lists
    score = sentence_bleu([ref_tokens], cand_tokens)
    return score

def main():
    csv_path = "test_cases.csv"  # Path to your CSV file containing test cases
    test_cases = load_test_cases(csv_path)
    total_cases = len(test_cases)
    total_bleu = 0.0
    results = []

    for case in test_cases:
        question = case['question']
        ground_truth = case['ground_truth']

        # If the question is in Sinhala, translate to English for processing.
        if any('\u0D80' <= char <= '\u0DFF' for char in question):
            english_query = sinhalaToEnglish(question)
        else:
            english_query = question

        # Retrieve documents and generate an answer using the LLM.
        docs = get_docs(english_query, top_k=5)
        generated_answer = generate_answer(english_query, docs)

        # If original question was Sinhala, translate generated answer back to Sinhala.
        if any('\u0D80' <= char <= '\u0DFF' for char in question):
            generated_answer = englishToSinhala(generated_answer)

        bleu = evaluate_bleu_score(ground_truth, generated_answer)
        total_bleu += bleu

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "bleu_score": bleu
        })

        print("Question:", question)
        print("Ground Truth:", ground_truth)
        print("Generated Answer:", generated_answer)
        print(f"BLEU Score: {bleu:.4f}")
        print("---")

    avg_bleu = total_bleu / total_cases if total_cases > 0 else 0.0
    print(f"Evaluated {total_cases} test cases. Average BLEU Score: {avg_bleu:.4f}")

    # Optionally, save the results to a CSV file for reporting.
    output_file = "evaluation_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question", "ground_truth", "generated_answer", "bleu_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()