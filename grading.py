import json
from collections import defaultdict, Counter

from datasets import load_dataset
from train.math_parsingutil import extract_answer, check_equivalence

#xx
def evaluate_pass_metrics_with_math500(json_file_path: str):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # First, examine the structure of a sample item
    if data:
        sample = data[0]
        print("Sample keys:", list(sample.keys()))
    
    grouped_by_pid = defaultdict(list)
    for row in data:
        pid = row["prompt_id"]
        grouped_by_pid[pid].append(row)

    total_questions = len(grouped_by_pid)
    pass_at_1_count = 0
    pass_at_8_count = 0
    majority_correct_count = 0

    for pid, attempts in grouped_by_pid.items():
        attempts = sorted(attempts, key=lambda x: x["sample_id"])
        
        # Check if any attempt has a ground truth answer
        if "instruction" in attempts[0]:
            print(f"Prompt ID: {pid}, Instruction: {attempts[0]['instruction']}")

        # Find the ground truth answer - look for the correct key
        gold_answer = None
        for key in ["ground_truth_answer", "answer", "gold_answer"]:
            if key in attempts[0]:
                gold_answer = attempts[0][key]
                break
        
        if not gold_answer:
            print(f"No gold answer found for prompt ID {pid}!")
            continue
            
        parsed_answers = [extract_answer(a.get("generated_answer", a.get("output", ""))) or "NO_ANSWER" for a in attempts]

        if check_equivalence(parsed_answers[0], gold_answer):
            pass_at_1_count += 1

        if any(check_equivalence(ans, gold_answer) for ans in parsed_answers):
            pass_at_8_count += 1

        # Rest of the code remains the same
        # ------ Majority Vote ------
        no_answer_count = sum(1 for ans in parsed_answers if ans == "NO_ANSWER")
        
        equivalent_groups = {}
        for ans in parsed_answers:
            if ans is None or ans == "NO_ANSWER":
                continue

            found_equiv = False
            for key in equivalent_groups:
                if check_equivalence(ans, key):
                    equivalent_groups[key].append(ans)
                    found_equiv = True
                    break
            if not found_equiv:
                equivalent_groups[ans] = [ans]

        max_count = no_answer_count
        majority_answer = "NO_ANSWER" if no_answer_count > 0 else None

        for key, group_answers in equivalent_groups.items():
            if len(group_answers) > max_count:
                max_count = len(group_answers)
                majority_answer = key

        if majority_answer is not None and check_equivalence(majority_answer, gold_answer):
            majority_correct_count += 1

    pass_at_1 = pass_at_1_count / total_questions
    pass_at_8 = pass_at_8_count / total_questions
    majority_acc = majority_correct_count / total_questions

    print("=============== EVALUATION ===============")
    print(f"Total questions:           {total_questions}")
    print(f"Pass@1 accuracy:           {pass_at_1:.3f}")
    print(f"Pass@8 accuracy:           {pass_at_8:.3f}")
    print(f"Majority voting accuracy:  {majority_acc:.3f}")
    print("==========================================")
if __name__ == "__main__":
    json_file = "llamadpo.json"
    evaluate_pass_metrics_with_math500(json_file)