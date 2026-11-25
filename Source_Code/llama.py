!pip install -q optimum-executorch optimum huggingface_hub
!python -m optimum.commands.optimum_cli export executorch \
  --model "unsloth/Llama-3.2-1B-Instruct" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --qlinear 8da4w \
  --qembedding 8w \
  --max_seq_len  \
  --output_dir "llama32_1b_pte" && zip -r llama32_1b_pte.zip llama32_1b_pte
