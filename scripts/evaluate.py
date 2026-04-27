import argparse
import json
import torch
import evaluate as hf_evaluate
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
from data.prepare_dataset import prepare

perplexity_metric = hf_evaluate.load("perplexity", module_type="metric")

def compute_perplexity(model, tokenizer, texts, batch_size=8):
    results = perplexity_metric.compute(
        model_id=None,
        predictions=texts,
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
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses

def evaluate_model(model_path, dataset, n_samples=200, model_name_base=None):
    base = model_name_base or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    test = dataset["test"].select(range(n_samples))
    prompts = test["prompt"]
    references = test["chosen"]

    print(f"Generating {n_samples} responses...")
    predictions = generate_responses(model, tokenizer, prompts)

    ppl = compute_perplexity(model, tokenizer, references)
    bs_f1 = compute_bertscore(predictions, references)

    return {"perplexity": ppl, "bertscore_f1": bs_f1, "n_samples": n_samples}

def main(args):
    dataset = prepare()
    results = {}

    for label, path in [
        ("zero_shot", args.base_model),
        ("sft", args.sft_path),
        ("dpo", args.dpo_path),
    ]:
        print(f"\n--- Evaluating: {label} ---")
        results[label] = evaluate_model(path, dataset, n_samples=args.n_samples, model_name_base=args.base_model)
        print(json.dumps(results[label], indent=2))

    output_path = "results/tables/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft_path", default="./models/checkpoints/sft")
    parser.add_argument("--dpo_path", default="./models/checkpoints/dpo")
    parser.add_argument("--n_samples", type=int, default=200)
    main(parser.parse_args())
