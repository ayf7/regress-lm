import json, random, numpy as np
import ioh

# Choose dimensions and how many task-instances per (function, dim)
dims = [2]
function_ids = list(range(1, 10))     # 1..24 BBOB functions
instances = list(range(1, 21))        # 50 per (func, dim) -> adjust as you like

# Per-task sampling budget  (small in-study train; bigger test)
N_train_per_task = 128
N_test_per_task  = 64
rng = np.random.default_rng(0)

def format_x_kv(x):
    # key-value text: x0:0.12,x1:-3.4,...
    return ",".join([f"x{i}:{float(v):.6g}" for i, v in enumerate(x)])

def task_metadata(problem):
    md = problem.meta_data
    return {
        "function": md.name,          # e.g., "RosenbrockRotated"
        "dim": md.n_variables,        # dimension
        "instance": md.instance       # instance id
    }

def sample_uniform(bounds, n, rng):
    lb, ub = np.array(bounds.lb), np.array(bounds.ub)
    return lb + rng.random((n, lb.size)) * (ub - lb)

TRAIN_PATH  = "bbob_shifted_train.jsonl"
TEST_PATH   = "bbob_shifted_test.jsonl"

with open(TRAIN_PATH, "w") as ftr, open(TEST_PATH, "w") as fte:
    for d in dims:
        for fid in function_ids:
            # for each (func, dim), pick several instances as separate tasks
            picked_instances = rng.choice(instances, size=10, replace=False)  # 10 tasks per (fid,d)
            for inst in picked_instances:
                prob = ioh.get_problem(fid, dimension=d, instance=int(inst))
                m = task_metadata(prob)

                # IOH gives bounds
                Xtr = sample_uniform(prob.bounds, N_train_per_task, rng)
                Xte = sample_uniform(prob.bounds, N_test_per_task, rng)

                for x in Xtr:
                    y = float(prob(x))               # evaluates f_k with that instance's transform
                    rec = {"x_text": format_x_kv(x), "m": m, "y": y}
                    ftr.write(json.dumps(rec) + "\n")

                for x in Xte:
                    y = float(prob(x))
                    rec = {"x_text": format_x_kv(x), "m": m, "y": y}
                    fte.write(json.dumps(rec) + "\n")

print("Wrote:", TRAIN_PATH, TEST_PATH)