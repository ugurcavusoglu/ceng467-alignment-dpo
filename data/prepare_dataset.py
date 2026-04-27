from datasets import load_dataset
import json
import os

def extract_prompt_and_response(sample):
    prompt = sample["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    chosen_response = sample["chosen"].rsplit("\n\nAssistant:", 1)[1].strip()
    rejected_response = sample["rejected"].rsplit("\n\nAssistant:", 1)[1].strip()
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}

def prepare(subset_size=None):
    print("Downloading Anthropic/hh-rlhf...")
    dataset = load_dataset("Anthropic/hh-rlhf")

    dataset = dataset.map(extract_prompt_and_response)
    dataset = dataset.filter(lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    dataset = dataset.filter(lambda x: x["chosen"] != x["rejected"])

    if subset_size:
        dataset["train"] = dataset["train"].select(range(min(subset_size, len(dataset["train"]))))

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples:  {len(dataset['test'])}")
    print("Sample:")
    print(json.dumps(dataset["train"][0], indent=2)[:500])
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Use N samples for fast dev iteration")
    args = parser.parse_args()
    prepare(subset_size=args.subset)
