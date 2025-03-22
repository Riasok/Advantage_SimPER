CUDA_VISIBLE_DEVICES="3"
accelerate launch \
   --config_file accelerate_config/fsdp_1gpu.yaml \         # accelerate config for 8-gpu allocation
   launch.py \                                              # main file for launching job
   loss=dpo \                                         # must be a file name in config/loss
   model=llama \                                            # must be a file name in config/model
   datasets=examples/binary_feedback.json                   # binary_feedback.json is a local file
   exp_name=llama3-8b_sft_dummy-kto \                       # experiment name, also the subfolder in cache dir for saving the model          
   ++cache_dir=/data1/home/minjaeoh/.cache/huggingface/hub \                               # set the cache directory 
   ++model.name_or_path=meta-llama/Llama-3.1-8B-Instruct \        # HF (or local) repo containing model configs, vocab, etc.
#    ++model.load_from=/data/models/llama3-8b_sft/FINAL/ \    # load existing model as starting point; if empty, use model.name_or_path
   ++lr=5e-6 \                                              # set the learning rate
   ++loss.beta=0.1     
   ++config.intermediate_checkpoints=true
   ++config.eval_every=500