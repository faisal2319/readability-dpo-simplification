"""Only generate GPT-4o outputs, then merge with existing Llama results."""
import json
from tqdm import tqdm
from openai import OpenAI

def extract_source(prompt):
    return prompt.split("easier to read:\n\n")[1].split("\n\nSimplified")[0].strip()

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

def main():
    # Check if we already have partial results
    try:
        with open("results/generations.json") as f:
            existing_results = json.load(f)
        print(f"Found {len(existing_results)} existing results")
    except FileNotFoundError:
        existing_results = []
    
    # Load test set
    with open("data/test_pairs.json") as f:
        test_pairs = json.load(f)
    sources = [extract_source(p["prompt"]) for p in test_pairs]
    
    # Generate GPT-4o only
    client = OpenAI()
    gpt4_results = []
    for src in tqdm(sources, desc="GPT-4o"):
        try:
            output = generate_gpt4(client, src)
            gpt4_results.append({"source": src, "model": "gpt4o", "output": output})
        except Exception as e:
            print(f"Error on source: {src[:50]}... — {e}")
    
    # Merge and save
    all_results = existing_results + gpt4_results
    with open("results/generations.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Total results: {len(all_results)}")

if __name__ == "__main__":
    main()