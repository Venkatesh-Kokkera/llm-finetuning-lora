import argparse
import mlflow
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load model with optional 4-bit quantization (QLoRA)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Loaded model with 4-bit quantization (QLoRA)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Loaded model in fp16")

    return model, tokenizer

def apply_lora(model, lora_r: int = 16, lora_alpha: int = 32):
    """Apply LoRA adapters to model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj",
            "v_proj", "o_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def format_instruction(sample: dict) -> str:
    """Format dataset sample as instruction."""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""

def train(args):
    """Main training function with MLflow tracking."""
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit
    )

    # Apply LoRA
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(
        "json",
        data_files={"train": args.dataset}
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # MLflow tracking
    mlflow.set_experiment("llm-finetuning-lora")

    with mlflow.start_run():
        mlflow.log_params({
            "model": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "use_4bit": args.use_4bit
        })

        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=tokenizer,
            formatting_func=format_instruction,
            max_seq_length=512
        )

        print("Starting training...")
        trainer.train()

        # Save adapter
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

        mlflow.log_artifact(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/train.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()
    train(args)
