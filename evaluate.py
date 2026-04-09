import json
import argparse
import mlflow
import nltk
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)

def load_finetuned_model(base_model: str, checkpoint: str):
    """Load fine-tuned model with LoRA adapters."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto"
    )

    print(f"Loading LoRA adapters from: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_length: int = 256
) -> str:
    """Generate response for a given instruction."""
    prompt = f"""### Instruction:
{instruction}

### Response:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()

def compute_bleu(references: list, hypotheses: list) -> float:
    """Compute corpus BLEU score."""
    refs = [[nltk.word_tokenize(r)] for r in references]
    hyps = [nltk.word_tokenize(h) for h in hypotheses]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smoothie)

def compute_rouge(references: list, hypotheses: list) -> dict:
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores.append(score["rougeL"].fmeasure)
    return {"rougeL": np.mean(scores)}

def compute_perplexity(
    model,
    tokenizer,
    texts: list
) -> float:
    """Compute average perplexity."""
    import torch
    perplexities = []

    for text in texts[:50]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexities.append(torch.exp(loss).item())

    return np.mean(perplexities)

def evaluate(args):
    """Run full evaluation pipeline."""
    # Load model
    model, tokenizer = load_finetuned_model(
        args.base_model,
        args.checkpoint
    )

    # Load test data
    print(f"Loading test data: {args.test_data}")
    with open(args.test_data, "r") as f:
        test_data = [json.loads(line) for line in f]

    print(f"Evaluating on {len(test_data)} samples...")

    references = []
    hypotheses = []

    for i, sample in enumerate(test_data[:100]):
        print(f"Sample {i+1}/100", end="\r")
        response = generate_response(
            model,
            tokenizer,
            sample["instruction"]
        )
        references.append(sample["output"])
        hypotheses.append(response)

    # Compute metrics
    bleu = compute_bleu(references, hypotheses)
    rouge = compute_rouge(references, hypotheses)
    perplexity = compute_perplexity(model, tokenizer, references)

    print(f"\n📊 Evaluation Results:")
    print(f"   BLEU Score  : {bleu:.4f}")
    print(f"   ROUGE-L     : {rouge['rougeL']:.4f}")
    print(f"   Perplexity  : {perplexity:.4f}")

    # Log to MLflow
    mlflow.set_experiment("llm-finetuning-lora")
    with mlflow.start_run():
        mlflow.log_metrics({
            "bleu_score": bleu,
            "rouge_l": rouge["rougeL"],
            "perplexity": perplexity
        })
        print("✅ Metrics logged to MLflow!")

    return {
        "bleu": bleu,
        "rouge_l": rouge["rougeL"],
        "perplexity": perplexity
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints"
    )
    parser.add_argument(
        "--test_data",
        default="data/processed/test.jsonl"
    )
    args = parser.parse_args()
    evaluate(args)
