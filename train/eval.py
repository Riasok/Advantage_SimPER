import json
import re
from collections import Counter
import sys
import os
# Import functions from the utils file
# Replace 'paste_2' with the actual module name without the .py extension
from train.math_parsingutil import *

def analyze_math_dataset(file_path):
    """
    Analyze math dataset to calculate accuracy metrics
    
    Args:
        file_path: Path to the file containing the JSON data
        
    Returns:
        tuple: (top1_accuracy, majority_accuracy, results_per_question)
    """
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse JSON objects
    json_pattern = r'(\{[\s\S]*?\}\s*),?'
    matches = re.findall(json_pattern, content)
    
    samples = []
    for match in matches:
        try:
            sample = json.loads(match)
            samples.append(sample)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON: {match[:100]}...")
    
    # Group samples by prompt_id
    questions = {}
    for sample in samples:
        prompt_id = sample.get('prompt_id')
        if prompt_id not in questions:
            questions[prompt_id] = []
        questions[prompt_id].append(sample)
    
    # Analyze results per question
    results = {}
    correct_top1 = 0
    correct_majority = 0
    total_questions = len(questions)
    
    for prompt_id, question_samples in questions.items():
        # Get ground truth answer and instruction
        ground_truth = question_samples[0].get('ground_truth_answer')
        instruction = question_samples[0].get('instruction')
        
        # Extract answers from generated text
        answers = []
        for sample in question_samples:
            generated_answer = sample.get('generated_answer')
            parsed_answer = extract_answer(generated_answer)
            answers.append(parsed_answer)
        
        # Check top-1 accuracy
        first_sample_correct = False
        if answers and check_equivalence(answers[0], ground_truth):
            correct_top1 += 1
            first_sample_correct = True
        
        # Check majority voting accuracy
        vote_counter = Counter(answers)
        if answers:
            majority_answer = vote_counter.most_common(1)[0][0]
            majority_correct = check_equivalence(majority_answer, ground_truth)
            if majority_correct:
                correct_majority += 1
        else:
            majority_correct = False
        
        # Store results for this question
        results[prompt_id] = {
            'instruction': instruction,
            'ground_truth': ground_truth,
            'answers': answers,
            'top1_correct': first_sample_correct,
            'majority_correct': majority_correct
        }
    
    # Calculate overall accuracy
    top1_accuracy = correct_top1 / total_questions if total_questions > 0 else 0
    majority_accuracy = correct_majority / total_questions if total_questions > 0 else 0
    
    return top1_accuracy, majority_accuracy, results

def print_results(top1_accuracy, majority_accuracy, results):
    """Print the analysis results"""
    print(f"\n{'=' * 60}")
    print(f"Math Dataset Accuracy Analysis")
    print(f"{'=' * 60}")
    
    print(f"\nTotal Questions: {len(results)}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Majority Voting Accuracy: {majority_accuracy:.4f} ({majority_accuracy*100:.2f}%)")
    
    print_detailed = input("\nDo you want to see detailed results for each question? (y/n): ")
    if print_detailed.lower() == 'y':
        print("\nDetailed Results:")
        print(f"{'-' * 60}")
        
        for prompt_id, result in results.items():
            print(f"Question ID: {prompt_id}")
            print(f"Instruction: {result['instruction'][:100]}..." if len(result['instruction']) > 100 else f"Instruction: {result['instruction']}")
            print(f"Ground Truth: {result['ground_truth']}")
            
            print("Extracted Answers:")
            for i, answer in enumerate(result['answers']):
                print(f"  Sample {i+1}: {answer}")
            
            print(f"Top-1 Correct: {result['top1_correct']}")
            print(f"Majority Correct: {result['majority_correct']}")
            print(f"{'-' * 60}")

def main():
    """Main function to run the analysis"""
    try:
        file_path = input("Enter the path to your data file: ")
        top1_accuracy, majority_accuracy, results = analyze_math_dataset(file_path)
        print_results(top1_accuracy, majority_accuracy, results)
        
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()