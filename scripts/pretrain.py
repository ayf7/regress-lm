from regress_lm.rlm import RegressLM
from regress_lm.core import Example
from typing import List
import random
import json

def row_to_example(row):
    # Prompt = <x_text> + <metadata text>
    m = row["m"]
    m_text = f"function:{m['function']},dim:{m['dim']},shift:{m['instance']}"
    prompt = f"{row['x_text']}\n{m_text}"
    return Example(x=prompt, y=float(row["y"]))

# Load a small subset first if you want to sanity check
def load_jsonl(path, limit=None) -> List[Example]:
    xs = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            xs.append(row_to_example(row))
            if limit and i+1>=limit: break
    return xs

print("loading data...")
train = load_jsonl("bbob_shifted_train.jsonl")
random.Random(0).shuffle(train)
valid = train[-int(0.1*len(train)):]
train = train[:-int(0.1*len(train))]
print(f"loaded {len(train)} train and {len(valid)} valid examples\n")

print("creating rlm...")
rlm = RegressLM.from_default(
        max_input_length=512,
        compile_model=False,
        batch_size=1024,
        batch_size_per_device=512,
    )
print("created rlm!\n")
print("fine tuning...")
rlm.fine_tune(train, validation_examples=valid, seed=0)