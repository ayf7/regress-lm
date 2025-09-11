from regress_lm.rlm import RegressLM
from regress_lm.core import Example
from typing import List
import random
import torch
import json
import os
from pathlib import Path

LOAD_FROM = None # will override SAVE_TO if set. If None, creates a new model.
SAVE_TO   = Path("runs/full/")
SAVE_FILE = None

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
# valid = train[0:1]
# train = train[1:2]

print(f"loaded {len(train)} train and {len(valid)} valid examples\n")

print("creating rlm...")


rlm = RegressLM.from_default(
        max_input_length=128,
        compile_model=False,
        batch_size=256,
        batch_size_per_device=256,
)

if LOAD_FROM:
    load_path = Path(LOAD_FROM)
    SAVE_TO = load_path.parent  # override save directory to match the loaded run
    print(f"loading checkpoint from {load_path} and saving future checkpoints to {SAVE_TO}")
    ckpt = torch.load(load_path, map_location="cpu")
    rlm.model.load_state_dict(ckpt["model_state_dict"])
    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state:
        try:
            rlm.fine_tuner.optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"[warn] optimizer state not loaded: {e}")

os.makedirs(SAVE_TO, exist_ok=True)
SAVE_FILE = SAVE_TO / SAVE_FILE if SAVE_FILE else SAVE_TO / "best.pt"
print("created rlm!\n")

print("fine tuning...")
try:
    rlm.fine_tune(train, validation_examples=valid, seed=0)
finally:
    torch.save({
        "model_state_dict": rlm.model.state_dict(),
        "optimizer_state_dict": rlm.fine_tuner.optimizer.state_dict(),
    }, SAVE_TO / "last.pt")

    print("saved checkpoint to", SAVE_TO / SAVE_FILE)