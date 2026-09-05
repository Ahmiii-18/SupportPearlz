import json
import os
from src.chains.rag_chain import get_rag_chain
from src.utils.logging_setup import logger

TEST_QUESTIONS_PATH = "evaluation/test_questions.json"
RESULTS_PATH = "evaluation/results/eval_results.json"

def run_evaluation():
    logger.info("Initializing automated evaluation benchmark...")
    
    if not os.path.exists(TEST_QUESTIONS_PATH):
        logger.error(f"Test suite file not found at {TEST_QUESTIONS_PATH}")
        return

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag_chain = get_rag_chain()
    results = []

    for case in test_cases:
        question = case.get("question")
        expected_in_domain = case.get("in_domain", True)
        
        logger.info(f"Testing Question: '{question}'")
        response = rag_chain.invoke(question)
        
        eval_record = {
            "id": case.get("id"),
            "question": question,
            "expected_in_domain": expected_in_domain,
            "response_answer": response.answer,
            "response_sources": response.sources,
            "confidence": response.confidence
        }
        results.append(eval_record)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()