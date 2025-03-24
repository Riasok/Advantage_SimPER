# from vllm import LLM, SamplingParams

# # Specify the full path to your model
# custom_model_path = "llama3-dpo-01-D-1e-5/FINAL"  # Update with your actual path
# base_model = "meta-llama/Llama-3-8B"  # Example base model

# # Ifrom vllm import LLM, SamplingParams
# # Initialize LLM with your fully merged model directly
# model = LLM(
#     model=custom_model_path,  # Directly load your merged model here
#     trust_remote_code=True,
#     dtype="half",  # or "bfloat16" depending on your hardware
# )

# # Define sampling parameters
# sampling_params = SamplingParams(
#     temperature=0.7,         # Controls randomness (higher = more random)
#     top_p=0.95,              # Nucleus sampling parameter
#     max_tokens=512           # Maximum number of tokens to generate
# )

# # Example prompt
# prompt = "Write a poem about artificial intelligence."

# # Generate text
# outputs = model.generate([prompt], sampling_params)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model using the full checkpoint directory
checkpoint_path = "/data1/home/minjaeoh/.cache/huggingface/hub/llama3-dpo-01-D-1e-5/FINAL"
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

# Save it as a single model file in a new directory
output_path = "/data1/home/minjaeoh/reasoning/HALOs/single-model"
model.save_pretrained(output_path)
AutoTokenizer.from_pretrained(checkpoint_path).save_pretrained(output_path)