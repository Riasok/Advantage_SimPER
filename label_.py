import argparse
import json
import os
import torch
import random
from tqdm import tqdm
from typing import List, Dict
import re
import sys
from train.dataloader import SFTDataLoader
from collections import defaultdict
from train.math_parsingutil import *

def format_output(output_str):
    return [{"role": "assistant", "content": output_str}]

def convert_to_binary_feedback(samples, seed = 42, fraction_test = 0.05):
    """
    Convert samples to binary feedback format. A sample is considered desirable if its
    reward crosses the threshold (label = 1) and undesirable otherwise (label = 0).
    
    The reward for a sample is set to 1 if the answer extracted from the output (via extract_answer)
    is equal to the sample's 'answer', and 0 otherwise.
    """
    # Group samples by prompt key
    grouped_by_prompt = defaultdict(list)
    
    for sample in samples:
        # Extract answer from sample's output
        extracted_ans = extract_answer(sample['output'])
        # Compare the extracted answer to the ground truth answer provided in the sample
        reward = 1 if extracted_ans == sample.get("answer") else 0
        prompt_key = ' '.join([x['content'] for x in sample['prompt']])
        
        sample['prompt_key'] = prompt_key
        sample['reward'] = reward
        sample['extracted_answer'] = extracted_ans
        
        grouped_by_prompt[prompt_key].append(sample)
    
    feedback = []
    for prompt_key, group in grouped_by_prompt.items():
        for sample in group:
            feedback_item = {
                'prompt_key': prompt_key,
                'prompt': sample['prompt'],
                'output': format_output(sample['output']),
                'label': 1 if sample['reward'] > 0 else 0,
                'reward': sample['reward'],
                'type': 'binary_feedback',
            }
            feedback.append(feedback_item)
    
    unique_prompt_keys = list(set(item['prompt_key'] for item in feedback))
    rnd = random.Random(seed)
    rnd.shuffle(unique_prompt_keys)
    n_test = int(fraction_test * len(unique_prompt_keys))
    test_prompt_keys = set(unique_prompt_keys[:n_test])
    
    # 각 피드백 항목에 split 정보를 추가하고 prompt_key는 제거
    for item in feedback:
        item['split'] = 'test' if item['prompt_key'] in test_prompt_keys else 'train'
        item.pop('prompt_key', None)

    return feedback

def convert_to_advantage_feedback(samples, seed = 42, fraction_test = 0.05):
    """
    Convert samples to advantage feedback format. All samples are labeled as desirable (label = 1),
    but they have different advantage scores based on how their correctness compares to the average.
    
    The advantage is calculated as: (r - mean(r_i)) / std(r_i) where r_i are all rewards for the same prompt.
    Samples with an advantage of 0 are dropped.
    """
    # Group samples by prompt key
    grouped_by_prompt = defaultdict(list)
    
    # First process all samples to extract answers and calculate rewards
    for sample in samples:
        extracted_ans = extract_answer(sample['output'])
        reward = 1 if extracted_ans == sample.get("answer") else 0
        prompt_key = ' '.join([x['content'] for x in sample['prompt']])
        
        sample['prompt_key'] = prompt_key
        sample['reward'] = reward
        sample['extracted_answer'] = extracted_ans
        
        grouped_by_prompt[prompt_key].append(sample)
    
    # Now calculate advantage for each sample
    feedback = []
    for prompt_key, group in grouped_by_prompt.items():
        # Calculate mean reward for this prompt
        rewards = [s['reward'] for s in group]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        
        # Calculate standard deviation
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_dev = variance ** 0.5
        else:
            std_dev = 0
        
        # Avoid division by zero
        std_dev = max(std_dev, 1e-8)
        
        # Check if all samples have same advantage (i.e., all have the same reward value)
        # If all rewards are the same, the advantage will be 0 for all of them
        all_same = all(r == rewards[0] for r in rewards)
        
        # If all advantages would be 0, skip this entire group
        if all_same:
            print(f"Skipping group with prompt key '{prompt_key}' - all samples have the same reward")
            continue
        
        for sample in group:
            # Calculate advantage: (r - mean(r)) / std(r)
            advantage = (sample['reward'] - mean_reward) / std_dev
            
            # Skip samples with advantage of 0
            if abs(advantage) < 1e-8:
                continue
                
            feedback_item = {
                'prompt_key': prompt_key,
                'prompt': sample['prompt'],
                'output': format_output(sample['output']),
                'label': 1,  # All samples are labeled as desirable
                'reward': sample['reward'],
                'advantage': advantage,
                'type': 'binary_feedback',
            }
            feedback.append(feedback_item)
    
    unique_prompt_keys = list(set(item['prompt_key'] for item in feedback))
    rnd = random.Random(seed)
    rnd.shuffle(unique_prompt_keys)
    n_test = int(fraction_test * len(unique_prompt_keys))
    test_prompt_keys = set(unique_prompt_keys[:n_test])
    
    # Add split information and remove prompt key
    for item in feedback:
        item['split'] = 'test' if item['prompt_key'] in test_prompt_keys else 'train'
        item.pop('prompt_key', None)

    print(f"Created {len(feedback)} advantage feedback samples after filtering out samples with advantage = 0")
    return feedback

def convert_to_pairwise_feedback(samples: List[Dict], seed: int = 42, fraction_test = 0.05) -> List[Dict]:
    """
    1) sample['prompt'] (리스트)를 합쳐서 prompt_key로 사용
    2) extract_answer()로 정답 여부 계산 -> sample['reward'] = 1(정답) or 0(오답)
    3) 동일 prompt_key 내의 정답/오답을 pairwise로 묶어 반환
    """
    random.seed(seed)

    # 1) prompt를 합쳐 prompt_key 생성 & reward 계산
    for sample in samples:
        # prompt가 리스트로 되어 있고,
        # 각 요소는 예: {"role": "user", "content": "..."} 형태라고 가정
        # content만 join하여 하나의 문자열로 만듦
        prompt_list = sample.get('prompt', [])
        prompt_texts = [p['content'] for p in prompt_list if 'content' in p]
        prompt_key = " ".join(prompt_texts).strip()

        sample['prompt_key'] = prompt_key if prompt_key else "no_prompt"

        extracted_ans = extract_answer(sample['output'])
        reward = 1 if extracted_ans == sample.get('answer') else 0
        sample['reward'] = reward

    # 2) prompt_key로 그룹화
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['prompt_key']].append(sample)

    pairwise_feedback = []

    # 3) 그룹별로 pairwise 만들기
    for prompt_key, group in grouped.items():
        correct_list = [s for s in group if s['reward'] == 1]
        incorrect_list = [s for s in group if s['reward'] == 0]

        c = len(correct_list)
        i = len(incorrect_list)

        # 전부 정답이거나 전부 오답이면 스킵
        if c == 0 or i == 0:
            continue

        # 페어 개수 = max(c, i)
        n_pairs = max(c, i)

        # 정답이 많거나 같은 경우
        if c >= i:
            for j in range(c):
                if j >= n_pairs:
                    break
                correct_sample = correct_list[j]
                # 오답은 필요한 만큼 반복
                incorrect_sample = incorrect_list[j % i]

                # A/B 순서를 랜덤하게 뒤집어 label 결정
                if random.random() < 0.5:
                    sample_A, sample_B = correct_sample, incorrect_sample
                    label = 1
                else:
                    sample_A, sample_B = incorrect_sample, correct_sample
                    label = 0

                feedback_item = {
                    'prompt_key': prompt_key,       
                    'prompt': sample_A.get('prompt', []),
                    'output_A': format_output(sample_A['output']),
                    'output_B': format_output(sample_B['output']),
                    'label': label,
                    'reward_A': sample_A['reward'],
                    'reward_B': sample_B['reward'],
                    'reward_difference': abs(sample_A['reward'] - sample_B['reward']),
                    'type': 'pairwise_feedback'
                }
                pairwise_feedback.append(feedback_item)

        # 오답이 더 많은 경우
        else:
            for j in range(i):
                if j >= n_pairs:
                    break
                incorrect_sample = incorrect_list[j]
                # 정답은 필요한 만큼 반복
                correct_sample = correct_list[j % c]

                # A/B 순서를 랜덤하게 뒤집어 label 결정
                if random.random() < 0.5:
                    sample_A, sample_B = correct_sample, incorrect_sample
                    label = 1
                else:
                    sample_A, sample_B = incorrect_sample, correct_sample
                    label = 0

                feedback_item = {
                    'prompt_key': prompt_key,       
                    'prompt': sample_A.get('prompt', []),
                    'output_A': format_output(sample_A['output']),
                    'output_B': format_output(sample_B['output']),
                    'label': label,
                    'reward_A': sample_A['reward'],
                    'reward_B': sample_B['reward'],
                    'reward_difference': abs(sample_A['reward'] - sample_B['reward']),
                    'type': 'pairwise_feedback'
                }
                pairwise_feedback.append(feedback_item)

    unique_prompt_keys = list(set(item['prompt_key'] for item in pairwise_feedback))
    rnd = random.Random(seed)
    rnd.shuffle(unique_prompt_keys)
    n_test = int(fraction_test * len(unique_prompt_keys))
    test_prompt_keys = set(unique_prompt_keys[:n_test])
    
    for item in pairwise_feedback:
        item['split'] = 'test' if item['prompt_key'] in test_prompt_keys else 'train'
        item.pop('prompt_key', None)

    return pairwise_feedback


def generate_accuracy_report(samples: List[Dict]) -> Dict:
    """
    Generate a comprehensive accuracy report for the samples.
    
    Returns:
    Dict with the following metrics:
    - pass_at_n: Whether any answer was correct per problem
    - pass_at_1: Whether the first answer was correct per problem
    - overall_accuracy: Percentage of correct answers out of all
    - majority_vote: Whether the majority of answers were correct per problem
    """
    # Group samples by prompt
    grouped_by_prompt = defaultdict(list)
    
    for sample in samples:
        prompt_list = sample.get('prompt', [])
        prompt_texts = [p['content'] for p in prompt_list if 'content' in p]
        prompt_key = " ".join(prompt_texts).strip()
        
        # Get or calculate reward
        if 'reward' in sample:
            reward = sample['reward']
        else:
            extracted_ans = extract_answer(sample['output'])
            sample['extracted_answer'] = extracted_ans
            reward = 1 if extracted_ans == sample.get('answer') else 0
            
        # Add reward to sample if not there
        if 'reward' not in sample:
            sample['reward'] = reward
            
        grouped_by_prompt[prompt_key].append(sample)
    
    # Initialize metrics
    total_problems = len(grouped_by_prompt)
    problems_with_correct_answer = 0
    problems_with_first_correct = 0
    problems_with_majority_correct = 0
    total_answers = 0
    total_correct_answers = 0
    
    # Calculate metrics
    for prompt_key, samples_for_prompt in grouped_by_prompt.items():
        # Sort samples by their order (if any order information is available)
        # For now, assume they're already in order
        
        any_correct = any(sample['reward'] == 1 for sample in samples_for_prompt)
        first_correct = samples_for_prompt[0]['reward'] == 1 if samples_for_prompt else False
        
        # Majority voting calculation - find most common answer
        extracted_answers = {}
        for sample in samples_for_prompt:
            # Extract the answer from the output if not already done
            if 'extracted_answer' not in sample:
                sample['extracted_answer'] = extract_answer(sample['output'])
            
            # Count each unique answer
            answer = sample['extracted_answer']
            extracted_answers[answer] = extracted_answers.get(answer, 0) + 1
        
        # Find the most common answer (majority vote)
        majority_answer = None
        max_count = 0
        for answer, count in extracted_answers.items():
            if count > max_count:
                max_count = count
                majority_answer = answer
        
        # Check if majority answer is correct
        correct_answer = samples_for_prompt[0].get('answer')
        majority_correct = (majority_answer == correct_answer)
        
        if any_correct:
            problems_with_correct_answer += 1
        
        if first_correct:
            problems_with_first_correct += 1
            
        if majority_correct:
            problems_with_majority_correct += 1
            
        # Count total answers and correct answers
        total_answers += len(samples_for_prompt)
        total_correct_answers += sum(sample['reward'] for sample in samples_for_prompt)
    
    # Calculate percentages
    pass_at_n_percent = (problems_with_correct_answer / total_problems * 100) if total_problems > 0 else 0
    pass_at_1_percent = (problems_with_first_correct / total_problems * 100) if total_problems > 0 else 0
    majority_vote_percent = (problems_with_majority_correct / total_problems * 100) if total_problems > 0 else 0
    overall_accuracy_percent = (total_correct_answers / total_answers * 100) if total_answers > 0 else 0
    
    # Create report
    report = {
        "total_problems": total_problems,
        "total_answers": total_answers,
        "total_correct_answers": total_correct_answers,
        "pass_at_n": {
            "count": problems_with_correct_answer,
            "percentage": round(pass_at_n_percent, 2)
        },
        "pass_at_1": {
            "count": problems_with_first_correct,
            "percentage": round(pass_at_1_percent, 2)
        },
        "majority_vote": {
            "count": problems_with_majority_correct,
            "percentage": round(majority_vote_percent, 2)
        },
        "overall_accuracy": {
            "percentage": round(overall_accuracy_percent, 2)
        }
    }
    
    return report


def main():
    if len(sys.argv) < 3:
        print("Usage: python label.py <input_file.json> <feedback_type: binary|pairwise|advantage>")
        sys.exit(1)

    input_file = sys.argv[1]
    feedback_type = sys.argv[2].lower()

    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if feedback_type == "binary":
        feedback = convert_to_binary_feedback(samples)
        output_file = input_file.replace(".json", "_binary_feedback.json")
    elif feedback_type == "pairwise":
        feedback = convert_to_pairwise_feedback(samples)
        output_file = input_file.replace(".json", "_pairwise_feedback.json")
    elif feedback_type == "advantage":
        feedback = convert_to_advantage_feedback(samples)
        output_file = input_file.replace(".json", "_advantage_feedback.json")
    else:
        print("Invalid feedback type. Choose either 'binary', 'pairwise', or 'advantage'.")
        sys.exit(1)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    output_score = input_file.replace(".json","_score.json")
    results = generate_accuracy_report(samples)
    with open(output_score, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Feedback saved to {output_file}")
    print(f"Scores saved to {output_score}")

if __name__ == "__main__":
    main()