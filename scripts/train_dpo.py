import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.prepare_dataset import prepare

def main(args):
    dataset = prepare(subset_size=args.subset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # DPO trainable model: SFT checkpoint + fresh LoRA
    model = AutoModelForCausalLM.from_pretrained(args.ref_model_path)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # PEFT mode: base weights act as ref
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=DPOConfig(
            beta=args.beta,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=50,
            save_strategy="epoch",
            report_to="wandb" if args.wandb else "none",
            max_length=512,
            max_prompt_length=256,
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"DPO model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--ref_model_path", default="./models/checkpoints/sft")
    parser.add_argument("--output_dir", default="./models/checkpoints/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    main(parser.parse_args())
