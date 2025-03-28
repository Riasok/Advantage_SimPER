"""
A script for sampling from LLMs. It should be run like this:

python -m train.sample /models/llama3-8B-sft/FINAL --output_file outputs.json \ 
    --gpu_count 4 --datasets alpacaeval --num_samples_per_prompt 4

The resulting JSON file with have items with the following fields:

- prompt: a list of key-value pairs with speaker and content
- instruction: only for one-turn eval datasets like alpacaeval
- output: clean output, without the chat template
- generator: path to either local model dir or Huggingface repo
- dataset: specific dataset that the prompt is from 
- split: either 'train' or 'test'
- prompt_id: unique integer for the prompt
- sample_id: integer from 0 to k - 1 for one of the k samples produced per prompt_id

The (prompt_id, sample_id) pair uniquely identifies each entry.

Note that the keys 'instruction', 'output' are necessary to run Alpacaeval on the samples.
"""
import argparse
import re
import os
import inspect
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from train.dataloader import SFTDataLoader
from train.utils import set_offline_if_needed
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import train.dataloader as dataloader_module
from .utils import StreamingJSONWriter

class PromptCreator:
    def __init__(self, tokenizer, system_prompt):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def create_prompt(self, question):
        if isinstance(question, list):
            if len(question) > 0 and isinstance(question[0], dict):
                question = " ".join(item.get("content", "") for item in question)
            else:
                question = " ".join(question)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                "Solve this math problem step by step. At the end, make sure to finish the calculation and state the answer exactly once in the following format: \n"
                "\"The final answer is \\boxed{X}\"\n, where X is your final answer.\n\n Q:" + question
            )}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
    
def get_available_datasets():
    """Get list of available datasets by finding all get_* functions in dataloader.py"""
    return [name[4:] for name, _ in inspect.getmembers(dataloader_module, inspect.isfunction) 
            if name.startswith('get_')]


def validate_datasets(datasets):
    """Validate that all requested datasets have corresponding get_* functions"""
    available_datasets = get_available_datasets()
    invalid_datasets = [d for d in datasets if d not in available_datasets]
    
    if invalid_datasets:
        available_str = "\n- ".join(available_datasets)
        raise ValueError(
            f"The following datasets are not available: {invalid_datasets}\n"
            f"Available datasets must have a corresponding get_* function in dataloader.py.\n"
            f"Currently available datasets are:\n- {available_str}"
        )

def main(args):
    validate_datasets(args.datasets)
    set_offline_if_needed()
    
    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    stop_tokens = [
            "</s>", "Question:", "Question", "USER:", "USER",
            "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
            "Response:", "Response"
        ]
    tokenizer.chat_template = open('config/template.jinja').read()
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        n=args.num_samples_per_prompt
    )
    system_prompt = "You are a helpful math assistant that solves problems step by step."
    prompt_creator = PromptCreator(tokenizer, system_prompt)

    prompt_idx = 0
    # Open the output file and create a streaming writer
    with open(args.output_file, 'w') as f:
        writer = StreamingJSONWriter(f)

        # Process each dataset
        for dataset in args.datasets:
            print(f"\nProcessing dataset: {dataset}")

            # Initialize the SFTDataLoader
            dataloader = SFTDataLoader(
                dataset_names=[dataset],
                tokenizer=tokenizer,
                split=args.split,
                max_prompt_length=args.max_prompt_length,
                n_epochs=1,
                seed=args.seed,
                microbatch_size=args.batch_size
            )

            # Process the dataset in batches
            for batch in dataloader:
                formatted_prompts = [prompt_creator.create_prompt(question) for question in batch['prompt']]
                responses = llm.generate(formatted_prompts, sampling_params)

                # Process and write each output
                for prompt, answer, response in zip(batch['prompt'], batch['answer'], responses):
                    for sample_idx, sample in enumerate(response.outputs):
                        cleaned_output = sample.text.strip()
                        cleaned_output = re.sub(r"<\|im_start\|>.*?\n", "", cleaned_output)  # role line 제거
                        cleaned_output = re.sub(r"<\|im_end\|>", "", cleaned_output)         # im_end 제거
                        cleaned_output = cleaned_output.strip()
                        output = {
                            "output": cleaned_output,
                            "generator": args.model_path,
                            "dataset": f"{dataset}_{args.split}",
                            "prompt_id": prompt_idx,
                            "sample_id": sample_idx,
                            "type": "sample",
                            "answer": answer,
                            "prompt": prompt
                        }

                        writer.write_item(output)

                    prompt_idx += 1

        writer.close()

    destroy_model_parallel()
    destroy_distributed_environment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--datasets", type=str, nargs="+", default=["alpacaeval"], help="List of datasets to sample from (space-separated)")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples to generate per input")
    parser.add_argument("--mode", type=str, default="alpacaeval", help="mode")
  
    args = parser.parse_args()
    main(args)