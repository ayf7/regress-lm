from regress_lm import rlm, core
import json, numpy as np

# Build a small model just to use its tokenizer
reg = rlm.RegressLM.from_default(max_input_len=8192)  # just for encoding stats

def prompt_of(row):
    m = row["m"]
    return f"{row['x_text']}\nfunction:{m['function']},dim:{m['dim']},shift:{m['instance']}"

lens = []
longest = []
for path in ["bbob_shifted_train.jsonl", "bbob_shifted_test.jsonl"]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            p = prompt_of(row)
            tok = reg.model.encoder_vocab.to_token_ids(p)            # internal tokenizer
            L = len(tok)
            lens.append(L)
            longest.append((L, p[:1200]))             # store a peek

lens = np.array(lens)
print("mean:", lens.mean(), "p95:", np.percentile(lens,95), "max:", lens.max())
print("Longest examples (top 3):")
for L, snippet in sorted(longest, reverse=True)[:3]:
    print(L, "tokens ::", snippet.replace("\n","  "))