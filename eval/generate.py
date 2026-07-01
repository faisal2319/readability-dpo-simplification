"""Generate simplifications from base, DPO-tuned, and GPT-4o models."""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from openai import OpenAI

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DPO_PATH = "Faisal2319/readability-dpo-llama32"
RESULTS_PATH = "results/generations.json"


def load_model(use_dpo=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    if use_dpo:
        model = PeftModel.from_pretrained(model, DPO_PATH)
    return model


def generate_llama(model, tokenizer, source):
    prompt = f"Simplify the following sentence so it is easier to read:\n\n{source}\n\nSimplified version:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("Simplified version:")[-1].strip()


def generate_gpt4(client, source):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You simplify English sentences for non-native speakers. Preserve meaning."},
            {"role": "user", "content": f"Simplify: {source}"}
        ],
        temperature=0.7,
        max_tokens=128
    )
    return response.choices[0].message.content.strip()


def extract_source(prompt):
    return prompt.split("easier to read:\n\n")[1].split("\n\nSimplified")[0].strip()


def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def already_done(results, model_name):
    return any(r["model"] == model_name for r in results)


def main():
    with open("data/test_pairs.json") as f:
        test_pairs = json.load(f)
    sources = [extract_source(p["prompt"]) for p in test_pairs]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    results = load_existing_results()
    print(f"Loaded {len(results)} existing results")

    # Base Llama
    if not already_done(results, "base_llama"):
        print("\n--- Base Llama ---")
        base = load_model(use_dpo=False)
        for src in tqdm(sources):
            results.append({
                "source": src,
                "model": "base_llama",
                "output": generate_llama(base, tokenizer, src)
            })
        save_results(results)
        print(f"Saved after base Llama: {len(results)} total")
        del base
        torch.cuda.empty_cache()
    else:
        print("Skipping base Llama (already done)")

    # DPO Llama
    if not already_done(results, "dpo_llama"):
        print("\n--- DPO Llama ---")
        dpo = load_model(use_dpo=True)
        for src in tqdm(sources):
            results.append({
                "source": src,
                "model": "dpo_llama",
                "output": generate_llama(dpo, tokenizer, src)
            })
        save_results(results)
        print(f"Saved after DPO Llama: {len(results)} total")
        del dpo
        torch.cuda.empty_cache()
    else:
        print("Skipping DPO Llama (already done)")

    # GPT-4o
    if not already_done(results, "gpt4o"):
        print("\n--- GPT-4o ---")
        client = OpenAI()
        for src in tqdm(sources):
            try:
                output = generate_gpt4(client, src)
                results.append({
                    "source": src,
                    "model": "gpt4o",
                    "output": output
                })
            except Exception as e:
                print(f"Error on source: {src[:50]}... — {e}")
        save_results(results)
        print(f"Saved after GPT-4o: {len(results)} total")
    else:
        print("Skipping GPT-4o (already done)")

    print(f"\nFinal total: {len(results)} generations")


if __name__ == "__main__":
    main()