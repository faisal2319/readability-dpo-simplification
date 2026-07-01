"""Build DPO preference pairs from ASSET dataset."""
import json
import textstat
from datasets import load_dataset
from tqdm import tqdm

def main():
    dataset = load_dataset("facebook/asset", "simplification", split="validation")
    preference_pairs = []
    
    for example in tqdm(dataset):
        source = example["original"]
        simplifications = example["simplifications"]
        
        if len(simplifications) < 2:
            continue
        
        scored = []
        for s in simplifications:
            if s.strip() and s != source:
                try:
                    fkgl = textstat.flesch_kincaid_grade(s)
                    scored.append((s, fkgl))
                except:
                    continue
        
        if len(scored) < 2:
            continue
        
        scored.sort(key=lambda x: x[1])
        
        # Only keep pairs with meaningful FKGL gap
        if scored[-1][1] - scored[0][1] < 2:
            continue
        
        preference_pairs.append({
            "prompt": f"Simplify the following sentence so it is easier to read:\n\n{source}\n\nSimplified version:",
            "chosen": scored[0][0],
            "rejected": scored[-1][0]
        })
    
    split_idx = int(len(preference_pairs) * 0.9)
    
    with open("data/train_pairs.json", "w") as f:
        json.dump(preference_pairs[:split_idx], f, indent=2)
    with open("data/test_pairs.json", "w") as f:
        json.dump(preference_pairs[split_idx:], f, indent=2)
    
    print(f"Train: {split_idx}, Test: {len(preference_pairs) - split_idx}")

if __name__ == "__main__":
    main()