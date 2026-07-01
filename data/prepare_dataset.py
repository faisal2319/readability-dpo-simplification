"""Build DPO preference pairs from ASSET dataset with composite readability score."""
import json
import textstat
from datasets import load_dataset
from tqdm import tqdm

def readability_score(text):
    """Composite readability score — lower is simpler."""
    try:
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return None
        
        # Average word syllables
        avg_syllables = textstat.avg_syllables_per_word(text)
        # Flesch-Kincaid grade
        fkgl = textstat.flesch_kincaid_grade(text)
        # Number of complex words (3+ syllables)
        difficult_words = textstat.difficult_words(text)
        difficult_ratio = difficult_words / n_words if n_words > 0 else 0
        
        # Composite: weighted sum. Lower = simpler.
        score = fkgl + (avg_syllables * 2) + (difficult_ratio * 10) + (n_words * 0.1)
        return score
    except:
        return None

def main():
    dataset = load_dataset("facebook/asset", "simplification", split="validation")
    preference_pairs = []
    skipped_short = 0
    skipped_similar = 0
    
    for example in tqdm(dataset):
        source = example["original"]
        simplifications = example["simplifications"]
        
        if len(simplifications) < 2:
            continue
        
        scored = []
        for s in simplifications:
            s_clean = s.strip()
            if not s_clean or s_clean == source:
                continue
            
            word_count = len(s_clean.split())
            source_word_count = len(source.split())
            if word_count < 5:
                skipped_short += 1
                continue
            if word_count < 0.3 * source_word_count:
                skipped_short += 1
                continue
            
            score = readability_score(s_clean)
            if score is not None:
                scored.append((s_clean, score))
        
        if len(scored) < 2:
            continue
        
        scored.sort(key=lambda x: x[1])
        
        # Require larger score gap
        if scored[-1][1] - scored[0][1] < 3:
            skipped_similar += 1
            continue
        
        # Additional check: chosen must not be substantially longer than rejected
        chosen_len = len(scored[0][0].split())
        rejected_len = len(scored[-1][0].split())
        if chosen_len > rejected_len * 1.2:
            skipped_similar += 1
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
    print(f"Skipped (too short): {skipped_short}")
    print(f"Skipped (similar readability): {skipped_similar}")

if __name__ == "__main__":
    main()