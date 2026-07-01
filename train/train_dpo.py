"""DPO fine-tuning of Llama 3.2 3B for readability control via QLoRA."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "/workspace/checkpoints/dpo-llama-readability"

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    with open("data/train_pairs.json") as f:
        train_pairs = json.load(f)
    train_dataset = Dataset.from_list(train_pairs)
    
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
        warmup_steps=50,
        report_to="none",
        bf16=True
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()