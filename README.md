🤖 LLM Domain Adaptation — Fine-Tuning LLaMA-2 & Mistral with LoRA/QLoRA

Parameter-efficient fine-tuning of open-source LLMs on domain-specific datasets — with BLEU, perplexity, and human feedback benchmarking before production deployment.

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

🎯 Problem
General-purpose LLMs underperform on specialized domains. Full fine-tuning costs $10K+ in GPU compute. This project uses LoRA and QLoRA to adapt LLaMA-2 and Mistral-7B to domain-specific tasks at a fraction of the cost — with a full evaluation pipeline before deployment.

✨ Features

LoRA + QLoRA — Train only 0.1% of parameters — 65% less GPU memory
Multi-Model — Adapters for LLaMA-2-7B, LLaMA-2-13B, and Mistral-7B
Instruction Tuning — Alpaca-style prompt formatting
Full Eval Pipeline — BLEU, ROUGE-L, perplexity, human preference scoring
MLflow Tracking — All experiments, metrics, and artifacts logged
Deployment Ready — Served via FastAPI on AWS SageMaker


📊 Results
ModelBLEU ScorePerplexityHuman PreferenceBase LLaMA-2-7B18.412.331%Fine-tuned LLaMA-2-7B31.77.168%Base Mistral-7B21.29.839%Fine-tuned Mistral-7B34.16.474%

🛠️ Tech Stack
ComponentTechnologyBase ModelsLLaMA-2-7B · LLaMA-2-13B · Mistral-7BFine-TuningPEFT — LoRA · QLoRAFrameworkPyTorch · HuggingFace TransformersEvaluationBLEU · ROUGE · PerplexityTrackingMLflowServingFastAPI · AWS SageMaker

🚀 Quick Start
bashgit clone https://github.com/Venkatesh-Kokkera/llm-finetuning-lora.git
cd llm-finetuning-lora
pip install -r requirements.txt
python train.py --model_name "meta-llama/Llama-2-7b-hf" --epochs 3
python evaluate.py --model_path ./checkpoints/

Venkatesh Kokkera · LinkedIn · Email
