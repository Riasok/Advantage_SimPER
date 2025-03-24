import os
import re
import json
import random
import torch
import numpy as np
import math_parsingutil
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class MathEvaluator:
    def __init__(self, model_name, output_dir, batch_size=4, seed=42):
        self.model_name = model_name
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        
        self.set_seed(self.seed)
        
        self.load_data()
        self.initialize_model_and_tokenizer()
        self.setup_sampling_params()
        
        self.all_results = []
        self.interesting_results = []
        self.no_answer_majority_results = []
        self.intermediate_metrics = []
        self.pass_at_1_correct = 0
        self.pass_at_64_correct = 0
        self.majority_correct = 0

        self.system_prompt = "You are a helpful math assistant that solves problems step by step."
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def load_data(self):
        subsets = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
        ]
        self.test_data  = load_dataset("HuggingFaceH4/MATH-500")
    
    def initialize_model_and_tokenizer(self):
        self.llm = LLM(self.model_name, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def setup_sampling_params(self):
        self.stop_tokens = [
            "</s>", "Question:", "Question", "USER:", "USER",
            "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
            "Response:", "Response"
        ]
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2048,
            stop=self.stop_tokens
        )
    
    def create_prompt(self, question):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                "Solve this math problem step by step. At the end, make sure to finish the calculation and state the answer exactly once in the following format: \n"
                "\"The final answer is \\boxed{X}\"\n, where X is your final answer.\n\n" + question
            )}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def process_batch(self, batch_examples, batch_idx):
        batch_prompts = []
        for example in batch_examples:
            question = example["problem"]
            formatted_prompt = self.create_prompt(question)
            batch_prompts.extend([formatted_prompt] * 16
                                 )
        
        batch_outputs = self.llm.generate(batch_prompts, sampling_params=self.sampling_params, use_tqdm=False)
        
        for i, example in enumerate(batch_examples):
            idx = batch_idx + i
            question = example["problem"]
            ground_truth = math_parsingutil.extract_answer(example["solution"])
            problem_type = example["type"]
            problem_level = example["level"]
            
            start_idx = i * 16
            end_idx = start_idx + 16
            example_outputs = batch_outputs[start_idx:end_idx]
            
            sample_results = []
            correct_count = 0
            
            for out in example_outputs:
                generated_text = out.outputs[0].text if out.outputs else ""
                extracted_answer = math_parsingutil.extract_answer(generated_text) or "NO_ANSWER"
                is_correct = math_parsingutil.check_equivalence(extracted_answer, ground_truth, question=question)
                correct_count += is_correct
                sample_results.append({
                    "question": question,
                    "type": problem_type,
                    "level": problem_level,
                    "ground_truth": ground_truth,
                    "model_answer": extracted_answer,
                    "trajectory": generated_text,
                    "is_correct": is_correct
                })
            
            # Majority voting count
            no_answer_count = sum(1 for r in sample_results if r["model_answer"] == "NO_ANSWER")
            equivalent_answers = {}
            for r in sample_results:
                if r["model_answer"] == "NO_ANSWER":
                    continue
                found_equivalent = False
                for key in equivalent_answers:
                    if math_parsingutil.is_equiv(r["model_answer"], key):
                        equivalent_answers[key].append(r["model_answer"])
                        found_equivalent = True
                        break
                if not found_equivalent:
                    equivalent_answers[r["model_answer"]] = [r["model_answer"]]
            
            max_count = no_answer_count  # "NO_ANSWER" number로 initialize
            vote_result = "NO_ANSWER" if no_answer_count > 0 else None
            vote_count = no_answer_count
            
            for key, values in equivalent_answers.items():
                if len(values) > max_count:
                    max_count = len(values)
                    vote_result = key
                    vote_count = len(values)
            
            if vote_result == "NO_ANSWER" and "NO_ANSWER" in [r["model_answer"] for r in sample_results]:
                vote_count = sum(1 for r in sample_results if r["model_answer"] == "NO_ANSWER")
            
            majority_is_correct = math_parsingutil.check_equivalence(vote_result, ground_truth, question=question)
            
            # update
            if majority_is_correct:
                self.majority_correct += 1
            if sample_results[0]["is_correct"]:
                self.pass_at_1_correct += 1
            if correct_count > 0:
                self.pass_at_64_correct += 1
            
            # majority vote가 틀렸으나 한 샘플 이상 정답인 경우 interesting result 저장
            if (not majority_is_correct) and (correct_count > 0):
                maj_candidates = [r for r in sample_results if math_parsingutil.is_equiv(r["model_answer"], vote_result)]
                maj_sample = random.choice(maj_candidates) if maj_candidates else None
                correct_candidates = [r for r in sample_results if r["is_correct"]]
                correct_sample = random.choice(correct_candidates) if correct_candidates else None

                interesting_result = {
                    "question_id": idx,
                    "question": question,
                    "type": problem_type,
                    "level": problem_level,
                    "ground_truth": ground_truth,
                    "majority_vote": {
                        "answer": vote_result,
                        "trajectory": maj_sample["trajectory"] if maj_sample else None,
                        "vote_count": vote_count
                    },
                    "correct_sample": {
                        "answer": correct_sample["model_answer"] if correct_sample else None,
                        "trajectory": correct_sample["trajectory"] if correct_sample else None,
                        "correct_count": correct_count 
                    }
                }
                self.interesting_results.append(interesting_result)
            
            # Majority vote가 NO_ANSWER인 경우 저장
            if vote_result == "NO_ANSWER":
                no_answer_candidates = [r for r in sample_results if r["model_answer"] == "NO_ANSWER"]
                no_answer_sample = no_answer_candidates[0] if no_answer_candidates else None
                no_ans_result = {
                    "question_id": idx,
                    "question": question,
                    "type": problem_type,
                    "level": problem_level,
                    "ground_truth": ground_truth,
                    "majority_vote": {
                        "answer": vote_result,
                        "trajectory": no_answer_sample["trajectory"] if no_answer_sample else None,
                        "vote_count": vote_count
                    }
                }
                self.no_answer_majority_results.append(no_ans_result)
            
            self.all_results.append({
                "question_id": idx,
                "question": question,
                "type": problem_type,
                "level": problem_level,
                "ground_truth": ground_truth,
                "samples": sample_results,
                "majority_vote": {
                    "answer": vote_result,
                    "count": vote_count,
                    "is_correct": majority_is_correct
                },
                "pass_at_1": sample_results[0]["is_correct"],
                "pass_at_32": (correct_count > 0),
                "correct_count": correct_count
            })
    
    def save_intermediate_results(self, current_count):
        """진행 중간 결과를 JSON 파일로 저장"""
        with open(os.path.join(self.output_dir, "Llama-3.1-8B_all_results.json"), "w") as f:
            json.dump(self.all_results, f, indent=2)
        with open(os.path.join(self.output_dir, "Llama-3.1-8B_interesting_results.json"), "w") as f:
            json.dump(self.interesting_results, f, indent=2)
        if current_count % 50 == 0 or current_count == len(self.test_data):
            intermediate_metric = {
                "questions_processed": current_count,
                "pass_at_1": self.pass_at_1_correct / current_count,
                "pass_at_32": self.pass_at_64_correct / current_count,
                "majority_voting": self.majority_correct / current_count,
                "total_examples": current_count
            }
            self.intermediate_metrics.append(intermediate_metric)
            with open(os.path.join(self.output_dir, "Llama-3.1-8B_metrics_intermediate.json"), "w") as f:
                json.dump(self.intermediate_metrics, f, indent=2)
    
    def save_final_results(self):
        total_examples = len(self.test_data)
        metrics = {
            "pass_at_1": self.pass_at_1_correct / total_examples,
            "pass_at_32": self.pass_at_64_correct / total_examples,
            "majority_voting": self.majority_correct / total_examples,
            "total_examples": total_examples
        }
        with open(os.path.join(self.output_dir, "Llama-3.1-8B_9b_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(self.output_dir, "Llama-3.1-8B_no_answer_majority.json"), "w") as f:
            json.dump(self.no_answer_majority_results, f, indent=2)
        return metrics
    
    def run_evaluation(self):
        total_examples = len(self.test_data)
        for batch_idx in tqdm(range(0, total_examples, self.batch_size)):
            batch_examples = self.test_data.select(range(batch_idx, min(batch_idx + self.batch_size, total_examples)))
            self.process_batch(batch_examples, batch_idx)
            current_count = min(batch_idx + self.batch_size, total_examples)
            self.save_intermediate_results(current_count)
        metrics = self.save_final_results()
        print(f"Evaluation completed. Results saved to {self.output_dir}")
        print(f"Pass@1: {metrics['pass_at_1']:.4f}")
        print(f"Pass@32: {metrics['pass_at_32']:.4f}")
        print(f"Majority Voting Accuracy: {metrics['majority_voting']:.4f}")


if __name__ == "__main__":
    evaluator = MathEvaluator(
        model_name="llama3-dpo-01-D-1e-5/FINAL",
        output_dir="Llama-3.1-8B_hendrycks_math_results_16_test",
        batch_size=4,
        seed=42
    )
    evaluator.run_evaluation()