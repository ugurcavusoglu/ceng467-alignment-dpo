import argparse
import json
import torch
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.prepare_dataset import prepare

import evaluate as hf_evaluate

def compute_perplexity(model, tokenizer, texts, batch_size=4):
    perplexity_metric = hf_evaluate.load("perplexity", module_type="metric")
    results = perplexity_metric.compute(
        predictions=texts,
        model_id=None,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return results["mean_perplexity"]

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return F1.mean().item()

def generate_responses(model, tokenizer, prompts, max_new_tokens=128):
    responses = []
    model.eval()
    device = next(model.parameters()).device
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses

def evaluate_model(model_path, dataset, n_samples=100, base_model_name=None):
    base = base_model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    test = dataset["test"].select(range(n_samples))
    prompts = list(test["prompt"])
    references = list(test["chosen"])

    print(f"  Generating {n_samples} responses...")
    predictions = generate_responses(model, tokenizer, prompts)

    print(f"  Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, references)

    print(f"  Computing BERTScore...")
    bs_f1 = compute_bertscore(predictions, references)

    del model
    torch.cuda.empty_cache()

    return {"perplexity": round(ppl, 4), "bertscore_f1": round(bs_f1, 4), "n_samples": n_samples}

def main(args):
    dataset = prepare()
    results = {}

    for label, path in [
        ("zero_shot", args.base_model),
        ("sft", args.sft_path),
        ("dpo", args.dpo_path),
    ]:
        print(f"\n--- Evaluating: {label} ---")
        results[label] = evaluate_model(path, dataset, n_samples=args.n_samples, base_model_name=args.base_model)
        print(json.dumps(results[label], indent=2))

    os.makedirs("results/tables", exist_ok=True)
    output_path = "results/tables/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft_path", default="./models/checkpoints/sft")
    parser.add_argument("--dpo_path", default="./models/checkpoints/dpo")
    parser.add_argument("--n_samples", type=int, default=100)
    main(parser.parse_args())
