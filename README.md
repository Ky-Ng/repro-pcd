# Reproducing `Predictive Concept Decoders` (PCD) on a Student's Budget

## Hotlinks 🌭
1. Walkthrough: [YouTube Playlist](https://www.youtube.com/playlist?list=PLd_GVe4IPmpAvb9zdWgl6QLMYOH-9TEkT)
2. Blogpost: [doingtheth.ing](https://doingtheth.ing/notes/projects/pcd/) (details the math behind PCDs)
3. Latest Training Run: [Wandb Pretraining Run](https://api.wandb.ai/links/kgng-usc/ed6mrtg9)
4. HuggingFace Artifacts: [kylelovesllms/hand-repro-pcd](https://huggingface.co/kylelovesllms/hand-repro-pcd/tree/main)
5. Original paper by Transluce: [Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants](https://arxiv.org/abs/2512.15712)

> Attempted 1 day hackathon to build Predictive Concept Decoders (PCDs) on a Student Budget turned into multi-month deep research side quest!

## Motivation
- I had the opportunity to see Sarah Schwettmann present Predictive Concept Decoders (PCDs) at the NeurIPS 2025 MechInterp Workshop! I didn't have the Interp understanding at the time to grasp the concepts (other than, *wow this is really cool!*)
- The goal of reproducing PCDs is two fold:
	1. Learn skills needed for SOTA Interp research by  struggling through the math and programming *by hand*
	2. Make research accessible to people starting out (like myself a few weeks ago!) who would like to learn about concepts like PCDs, sparsity bottlenecks, AutoInterp through a project which holds your hand through each step
- Hopefully fellow students who don't have a ton of compute on hand can find this as a useful guide into getting into AI Safety research, some theory (linear representation hypothesis, transformer architecture, sparsity), some practical aspects (GPU setup, details (like tokenizer padding and a bit of how  KV Cache/RoPE is affected when you slot in activations)), and repo architecture (`src`, `checkpoints`, uploading to Huggingface)
## Goal
1. Reproduce 5 phases of Predictive Concept Decoders by hand (write each line/understand what's happening to as much detail as I can!)
	1. ✅  Inference Pipeline (E2E with random weights)
	2. ✅ Pretraining Pipeline
	3. 📋 Encoder Feature Labelling using AutoInterp Methods
	4. 📋 SynthSys Dataset Generation
	5. 📋 FineTuning Pipeline
2. Create a video series on concepts/how to reproduce each step of the architecture
	1. Heavily inspired by Andrej Karpathy's [`Let's build GPT: from scratch, in code, spelled out.`](https://www.youtube.com/watch?v=kCc8FmEb1nY)
3. To Pareto Principle (80-20) this, we will skip the evals/baselines section. Assuming we've followed the paper correctly, we do not need to verify we're doing better than baselines (also this is more for educational purposes)
## Quick Start

### Download Repo
```zsh
git clone git@github.com:Ky-Ng/repro-pcd.git
```

### Pull Down Checkpoints
- Note: the repo where the checkpoints live is `kylelovesllms/hand-repro-pcd`

1. To pull down a specific checkpoint (e.g. PCD trained on 20K steps, at step 20,000)
```zsh
huggingface-cli download kylelovesllms/hand-repro-pcd \
    --include "checkpoints/steps_20K/step_20000/*" \
    --local-dir "./out/checkpoints/steps_20K/step_20000/*"
```

2. To pull down all checkpoints  
```zsh
huggingface-cli download kylelovesllms/hand-repro-pcd \
    --include "checkpoints/steps_20K/*" \
    --local-dir "./out/checkpoints/steps_20K"
```
### Run Evaluation

#### Evaluate the  End-to-End PCD Pipeline from Checkpoint
 - `prompt` is passed into the `Subject Model` --> Activations passed through the `Encoder` --> Sparse Activations passed as soft tokens to `Decoder Model`
 - `decoder_question` is passed in as hard tokens after the soft tokens with the chat template
	 - TODO: Figure out how the chat template should be applied for instruction tuned models (paper seems ambiguous about this)
```zsh
python -m evals.evaluate_prompt \
	--prompt="I am a vegan. I have an allergy to lentils." \
	--decoder_question="Please advise me on food to make." \
	--checkpoint="out/checkpoints/steps_20K"
```

#### Evaluate the `DecoderModel`
- Test the capabilities of the `DecoderModel` to have respond in chat template during each training phase (specifically, check after pretraining which only uses FineWeb completions)

- Evaluate on random weights
```zsh
python -m evals.evaluate_chat \
	--prompt="I am a vegan. I have an allergy to lentils. Please advise me on food to make." 
```

- Optionally specify a checkpoint weights for the decoder model
```zsh
python -m evals.evaluate_chat \
	--prompt="I am a vegan. I have an allergy to lentils. Please advise me on food to make." \
	--checkpoint="out/checkpoints/steps_20K"
```
### Run Tests

Each test is a runnable smoke-test script (not pytest) that exercises one component end-to-end on the GPU.

- `test_subject_model.py` — verifies the frozen subject model pipeline: chat-templating, tokenization, generation, and pulling residual-stream activations at `l_read` over the middle-token span.
```
python -m tests.test_subject_model
```

- `test_sparse_encoder.py` — verifies the sparse encoder forward pass (`W_enc` → `TopK` → `W_emb`) on random activations and prints the output shape along with the TopK info dict.
```
python -m tests.test_sparse_encoder
```

- `test_decoder_model.py` — verifies the LoRA-wrapped decoder: `forward_train` under both the FineWeb pretrain layout (soft tokens + suffix) and the SynthSys finetune layout (soft tokens + context + target), `generate` from random soft tokens, and the chat-template path with zero-masked dummy soft tokens.
```
python -m tests.test_decoder_model
```

## Repo Structure
<img width="3131" height="3248" alt="image" src="https://github.com/user-attachments/assets/7729cea5-cd13-4519-b972-c10c30693d24" />

### Code Structure
- Claude Code Generated Structure
```zsh
repro-pcd/
├── src/
│   ├── pcd_config.py                  # Dataclass with all hyperparameters (model, LoRA, training, dataset, paths)
│   │
│   ├── architecture/
│   │   ├── subject_model.py           # Frozen Qwen2.5-1.5B via TransformerLens; forward hook pulls residual-stream activations at l_read
│   │   ├── sparse_encoder.py          # Encoder/Decoder pair (W_enc → TopK → W_emb) that produces the sparse "soft tokens" and the aux loss for dead-concept revival
│   │   ├── decoder_model.py           # LoRA-wrapped Qwen instruct model; patches soft tokens in place of the prefix and handles forward_train / generate / chat-template
│   │   └── pcd_inference_model.py     # End-to-end glue: subject → encoder → decoder for deployment-time generation from a checkpoint
│   │
│   ├── data/
│   │   ├── fine_web_dataset.py        # Downloads + tokenizes FineWeb into fixed 48-token windows (n_prefix + n_middle + n_suffix); yields PyTorch DataLoader
│   │   └── upload_to_hf.py            # CLI helper to push checkpoints / data caches to a Hugging Face Hub repo
│   │
│   ├── training/
│   │   ├── train_pretraining.py       # Joint Encoder + Decoder-LoRA pretraining loop on FineWeb (next-token loss + L_aux, wandb logging, periodic checkpoints)
│   │   └── utils.py                   # save_checkpoint / load_checkpoint (encoder.pt + decoder_lora/adapter_model.safetensors) + log_metrics
│   │
│   └── entrypoints/
│       └── pretrain.py                # `python -m src.entrypoints.pretrain --run-name ...` CLI that calls training/train_pretraining.train
│
├── evals/
│   ├── evaluate_prompt.py             # Runs the full PCD pipeline (subject → encoder → decoder) on a user prompt from a checkpoint
│   └── evaluate_chat.py               # Probes whether chat-template / instruction-following survives FineWeb pretraining by feeding zero-masked dummy soft tokens + a chat-templated prompt
│
├── tests/
│   ├── test_subject_model.py          # Smoke-tests activation extraction at l_read and the chat-template path
│   ├── test_sparse_encoder.py         # Smoke-tests forward pass, TopK sparsity, and auxiliary-loss shapes
│   └── test_decoder_model.py          # Smoke-tests forward_train (pretrain + finetune layouts) and generate, including the chat-template path
│
├── out/                               # (gitignored) wandb runs, data_cache/, and checkpoints/<run-name>/step_<N>/{encoder.pt, decoder_lora/, optimizer.pt}
├── requirements.txt
└── README.md
```
