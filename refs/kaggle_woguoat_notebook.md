[woguoat's profile](https://www.kaggle.com/woguoat) woguoat  · 4d ago · 123 views

arrow\_drop\_up18

Copy & Edit
19

file\_downloadDownload

![bronze medal](https://www.kaggle.com/static/images/medals/competitions/bronzel@2x.png)

more\_vert

# ARC2 NVARC Full-Cost SOTA Triple Pass

## ARC2 NVARC Full-Cost SOTA Triple Pass

[Notebook](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass/notebook) [Input](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass/input) [Output](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass/output) [Logs](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass/log) [Comments (0)](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass/comments)

historyVersion 1 of 1

## Runtime

play\_arrow

28m 20s · GPU L4 x4

## Input

Competitions

![](https://www.kaggle.com/competitions/133469/images/thumbnail)

ARC Prize 2026 - ARC-AGI-2

Models

![](https://storage.googleapis.com/kaggle-avatars/images/502311-kg.png)

qwen3\_4b\_grids15\_sft139 · ...s15\_sft139 · V1

## Tags

[GPU](https://www.kaggle.com/code?tagIds=16580-GPU)

## Language

Python

## Table of Contents

[ARC-AGI-2: NVARC test-time training, full-depth single pass + pooled second pass](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass#ARC-AGI-2:-NVARC-test-time-training,-full-depth-single-pass-+-pooled-second-pass)

![Profile picture for undefined](https://storage.googleapis.com/kaggle-competitions/kaggle/133469/logos/thumb76_76.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1793130329&Signature=TThtSUJHtEeTd07TWOchTo%2FPHBLFGLBjd%2FLK%2Bm6fx%2BPnFPzVFXNdq%2FL7vlqDzr%2FTh5rgnrUQW0MPlG2ssUBrVj2ot%2Bk%2FhI2eXWw1r32%2FBJ8nAxuB8gy7uQp7RDbETDqhcUk16vIuvGJAVCR5cQ75xrbJVkILcV5zL%2BGTlqcJuveoTr%2FHkc%2BIn654lyRvqYVcgHflOlld%2BbcRq%2BXooHPWfoT5vsf7mJjW54CQLyb4j9cp1kesYN4rwdX7mI2Xo8IqEh%2Fm6epNpza7mqYL0qkHkapTUQTp8lbfBAvENTrgj0sBu2pe5%2FGADodvaeGDjEPONZZuySVKpSRm%2FkyNZGMy7A%3D%3D&t=2026-03-10-16-17-03)

Competition Notebook

[ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)

Public Score

31.94

Best Score

[31.94 V1](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass?scriptVersionId=361430924)

\_\_notebook\_\_

Loading \[MathJax\]/jax/output/CommonHTML/fonts/TeX/fontdata.js

linkcode

# ARC-AGI-2: NVARC test-time training, full-depth single pass + pooled second pass [¶](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass\#ARC-AGI-2:-NVARC-test-time-training,-full-depth-single-pass-+-pooled-second-pass)

The public NVARC Qwen3-4B pipeline with its model math unchanged (128 TTT sequences, LoRA r=256,
16 augmented views, turbo DFS at p>=0.2, 8-view augmentation rescoring, `score_kgmon` selection).

Changes are throughput and robustness only:

1. **Length-grouped DFS batches of up to 8 views** (views of one shape class share a token length);
on L4 an 8-row step costs about the same as a 4-row step, so decoding runs ~2x faster.
2. **Cheapest tasks first + per-task time budgets** from an online-calibrated cost model, so every
task is processed and each keeps the baseline's ~1200 s (+ running batch) allowance.
3. **Leftover time** re-runs the least certain tasks with fresh seeds (second TTT run, 16 views);
candidates are pooled with the first run under `score_kgmon`.
4. Stable crc32 rescoring seeds, per-task error isolation, corrupt-file tolerant submission writer.

In \[1\]:

linkcode

```
import os, time
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
T0 = time.time()
# Hidden-set run: 12 h limit minus a 15 min margin for the submission cell.
# Commit run: a short functional check on the 4 evaluation tasks used by the public notebook.
global_end_time = T0 + (12 * 3600 - 900 if RERUN else 32 * 60)
print("rerun:", RERUN, "budget (h):", round((global_end_time - T0) / 3600, 2))
```

```
rerun: False budget (h): 0.53
```

In \[2\]:

linkcode

```
# Preserve the baseline environment workaround.
!pip uninstall -y tensorflow
```

```
Found existing installation: tensorflow 2.18.0
Uninstalling tensorflow-2.18.0:
  Successfully uninstalled tensorflow-2.18.0
```

In \[3\]:

linkcode

```
%%writefile arc_loader.py
import json
import numpy as np
from transformers import AutoTokenizer

def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def is_valid_solution(guess):
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))

class QwenFormatter:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None

class ArcDataset:

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if   op=='rot90':              a = np.rot90(a, k=3)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = f.read()
        return cls(
            queries=json.loads(queries),
            is_orig=True,
            keys=keys,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
```

```
Writing arc_loader.py
```

In \[4\]:

linkcode

```
%%writefile arc_decoder.py
import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)

selection_algorithms = [\
    score_full_probmul_3,\
    score_kgmon,\
]

class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:  # a worker killed mid-write leaves a truncated file
                print(f"*** skipping unreadable result file {key}: {e!r}")
                continue
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
        print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
```

```
Writing arc_decoder.py
```

In \[5\]:

linkcode

```
%%writefile arc_sched.py
"""Cost model and budget pacing for the 4-worker TTT pipeline.

The cost model was fitted on 24 ARC-AGI-2 evaluation tasks run with the baseline
configuration (128 TTT sequences, 16 views, DFS p>=0.2) on one L4:
  train  ~ 19.2 + 0.0809*S + 4.16e-6*S^2   seconds per 128 steps (S = training sequence tokens)
  decode ~ 185 + 0.588*OT - 0.0381*OT*S/1000   seconds for 16 views (OT = predicted output tokens)
Training is predicted within ~10 s, decoding only within ~230 s, so the pacer calibrates
a global factor kappa = actual / predicted online and re-plans every task from the
remaining wall time.
"""
import time

import numpy as np

TRAIN_COEF = (19.2334, 0.0809330, 4.16404e-6)
DECODE_COEF = (185.248, 0.588264, -0.0381164)
SETUP_S = 15.0

# Level 0 = baseline config. The baseline stops starting decode batches 1200 s into a task
# and lets the running batch finish (up to its DFS window), so tasks can last ~1700 s; the
# cap reproduces that. Boost levels are disabled: the v8 run with them scored 26.39.
LEVELS = {
    0: dict(n_train_aug=16, extra_views=False, factor=1.00, cap=1700.0),
    1: dict(n_train_aug=16, extra_views=True,  factor=1.30, cap=1700.0),
    2: dict(n_train_aug=24, extra_views=True,  factor=1.55, cap=1900.0),
}
LEVEL1_RHO = float("inf")
LEVEL2_RHO = float("inf")
KAPPA_PRIOR_WEIGHT = 1500.0

PASS_SEEDS = {
    "A":  dict(train_seed=1, lora_seed=42,  eval_seed=2, suffix=""),
    "B":  dict(train_seed=17, lora_seed=137, eval_seed=29, suffix=".p2"),
}

def seq_tokens(task):
    return sum(len(g) * (len(g[0]) + 1) + 6 for p in task["train"] for g in (p["input"], p["output"]))

def predicted_output_tokens(task, test_item):
    ins = [np.shape(p["input"]) for p in task["train"]]
    outs = [np.shape(p["output"]) for p in task["train"]]
    x = np.shape(test_item["input"])
    if all(a == b for a, b in zip(ins, outs)):
        o = x
    elif len(set(outs)) == 1:
        o = outs[0]
    else:
        ry = np.mean([b[0] / a[0] for a, b in zip(ins, outs)])
        rx = np.mean([b[1] / a[1] for a, b in zip(ins, outs)])
        o = (max(1, min(30, round(x[0] * ry))), max(1, min(30, round(x[1] * rx))))
    return o[0] * (o[1] + 1) + 1

def est_train_seconds(task, n_train_aug=16):
    S = min(seq_tokens(task), 8192)
    per128 = TRAIN_COEF[0] + TRAIN_COEF[1] * S + TRAIN_COEF[2] * S * S
    return per128 * n_train_aug / 16

def est_decode_seconds(task):
    S = min(seq_tokens(task), 8192)
    total = 0.0
    for t in task["test"]:
        ot = predicted_output_tokens(task, t)
        total += DECODE_COEF[0] + DECODE_COEF[1] * ot + DECODE_COEF[2] * ot * S / 1000
    # the fit is per task (one test output for most tasks); extra outputs add less than a full task
    n = len(task["test"])
    total = total * (1 + 0.6 * (n - 1)) / n
    return float(min(max(total, 60.0), 1300.0))

def est_task_seconds(task):
    return SETUP_S + est_train_seconds(task) + est_decode_seconds(task)

def output_uncertainty(u):
    """0 = settled, higher = worth another independent TTT run."""
    if u["c1"] == 0:
        return 0.25  # no candidate at all: a second run rarely recovers these
    if u["c1"] >= max(6, u["views"] // 2) and u["c1"] >= 2 * u["c2"] + 2:
        return 0.0
    return 1.0 - u["c1"] / max(u["views"], 1) + 0.25 * min(u["n_unique"], 4) / 4

class Pacer:
    """Hands out tasks and per-task wall budgets to one worker.

    shared: dict of Manager proxies created by starter.py
      lock, order (list of keys, cheap first), est (key -> base seconds),
    tasks: key -> ARC task (train/test inputs), for the training-time floor
      state (qi, rem, act, exp), busy (rank -> busy-until ts),
      unc (key -> per-output uncertainty), tail_done (key -> 1), failed (key -> 1)
    """

    def __init__(self, shared, rank, n_workers, end_time, tasks):
        self.s = shared
        self.tasks = tasks
        self.rank = rank
        self.n_workers = n_workers
        self.end_time = end_time

    def kappas(self):
        st = self.s["state"]
        kt = (KAPPA_PRIOR_WEIGHT + st["tr_act"]) / (KAPPA_PRIOR_WEIGHT + st["tr_exp"])
        kd = (KAPPA_PRIOR_WEIGHT + st["dec_act"]) / (KAPPA_PRIOR_WEIGHT + st["dec_exp"])
        return kt, kd

    def kappa(self):
        """Overall actual/predicted ratio for a typical task mix."""
        kt, kd = self.kappas()
        st = self.s["state"]
        return (kt * st["rem_tr0"] + kd * st["rem_dec0"]) / max(st["rem_tr0"] + st["rem_dec0"], 1.0)

    def task_seconds(self, key, level=0):
        kt, kd = self.kappas()
        lv = LEVELS[level]
        t = self.tasks[key]
        dec_factor = 1.35 if lv["extra_views"] else 1.0
        return SETUP_S + kt * est_train_seconds(t, lv["n_train_aug"]) + kd * est_decode_seconds(t) * dec_factor

    def available(self, now):
        others = 0.0
        for r, until in self.s["busy"].items():
            if r != self.rank:
                others += max(0.0, until - now)
        return self.n_workers * max(0.0, self.end_time - now) - others

    def _plan(self, key, pass_name, rem_tr, rem_dec, now):
        kt, kd = self.kappas()
        A = self.available(now)
        task = self.tasks[key]
        base = self.task_seconds(key, 0)
        D = kt * rem_tr + kd * rem_dec + base
        rho = A / max(D, 1.0)
        if pass_name == "A":
            level = 2 if rho >= LEVEL2_RHO else (1 if rho >= LEVEL1_RHO else 0)
        else:
            level = 0
        lv = LEVELS[level]
        expected = self.task_seconds(key, level)
        train_floor = SETUP_S + kt * est_train_seconds(task, lv["n_train_aug"]) + 90.0
        if rho >= 1.0:
            # on schedule: give the task the full baseline allowance, never less
            budget = lv["cap"]
        else:
            # behind schedule: shrink this task's share in proportion
            budget = min(lv["cap"], 1.2 * base * max(rho, 0.25))
        budget = max(budget, train_floor)
        budget = min(budget, self.end_time - now - 20.0)
        plan = dict(task=key, pass_name=pass_name, level=level, est=self.s["est"][key], kappa=kd, rho=rho,
                    budget=budget, expected=min(budget, expected),
                    est_train=est_train_seconds(task, lv["n_train_aug"]), est_dec=est_decode_seconds(task) * (1.35 if lv["extra_views"] else 1.0),
                    n_train_aug=lv["n_train_aug"], extra_views=lv["extra_views"],
                    **PASS_SEEDS[pass_name])
        return plan, train_floor

    def next_task(self):
        while True:
            now = time.time()
            if now > self.end_time - 120:
                return None
            with self.s["lock"]:
                st = self.s["state"]
                qi = st["qi"]
                order = self.s["order"]
                if qi < len(order):
                    key = order[qi]
                    task = self.tasks[key]
                    st["qi"] = qi + 1
                    st["rem_tr0"] = st["rem_tr0"] - est_train_seconds(task)
                    st["rem_dec0"] = st["rem_dec0"] - est_decode_seconds(task)
                    plan, train_floor = self._plan(key, "A", st["rem_tr0"], st["rem_dec0"], now)
                    if now + train_floor > self.end_time - 20:
                        print(f"[Rank {self.rank}] skip {key}: not enough time to train "
                              f"(needs {train_floor:.0f}s, {self.end_time - now:.0f}s left)")
                        continue
                    self.s["busy"][self.rank] = now + plan["expected"]
                    return plan
                # main queue exhausted -> tail pass on the most uncertain finished tasks
                plan = self._pick_tail(now)
                if plan is None:
                    self.s["busy"][self.rank] = now
                    return None
                self.s["busy"][self.rank] = now + plan["expected"]
                return plan

    def _pick_tail(self, now):
        left = self.end_time - now - 30.0
        best, best_score = None, 0.0
        done = self.s["tail_done"]
        failed = dict(self.s["failed"])
        for key, unc in list(self.s["unc"].items()) + [(k, None) for k in failed]:
            if key in done:
                continue
            est = self.task_seconds(key, 0)
            if est * 1.1 > left:
                continue
            if unc is None:
                score = 2.0
            else:
                us = [output_uncertainty(u) for u in unc.values()]
                score = float(np.mean(us)) if us else 0.0
            if score <= 0.05:
                continue
            score = score / (est / 600.0 + 0.5)
            if score > best_score:
                best, best_score = key, score
        if best is None:
            return None
        done[best] = 1
        est = self.task_seconds(best, 0)
        pass_name = "A" if best in failed else "B"
        plan, _ = self._plan(best, pass_name, 0.0, 0.0, now)
        plan["budget"] = min(max(plan["budget"], est), left)
        plan["expected"] = min(plan["budget"], est)
        return plan

    def done(self, plan, actual, info):
        with self.s["lock"]:
            st = self.s["state"]
            if info is not None and "train_s" in info:
                st["tr_act"] = st["tr_act"] + max(info["train_s"] - SETUP_S, 1.0)
                st["tr_exp"] = st["tr_exp"] + plan["est_train"]
                dec = max(actual - info["train_s"], 1.0)
                if info.get("timed_out", False):
                    dec *= 1.3  # censored: the decode would have run longer
                st["dec_act"] = st["dec_act"] + dec
                st["dec_exp"] = st["dec_exp"] + plan["est_dec"]
            self.s["busy"][self.rank] = time.time()
            if info is None:
                if plan["pass_name"] == "A":
                    self.s["failed"][plan["task"]] = 1
            elif plan["pass_name"] == "A":
                self.s["unc"][plan["task"]] = info["unc"]
                if plan["task"] in self.s["failed"]:
                    del self.s["failed"][plan["task"]]
```

```
Writing arc_sched.py
```

In \[6\]:

linkcode

```
%%writefile arc_solver.py
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import json
import time
import zlib
import traceback
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

from arc_sched import Pacer

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15

def _env(name, default, cast):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

# Defaults reproduce the public NVARC baseline (128 TTT sequences, 16 views, DFS p>=0.2).
CFG = dict(
    model_path       = _env("ARC_MODEL_PATH", "", str),
    out_dir          = _env("ARC_OUT_DIR", "/kaggle/inference_outputs", str),
    telemetry        = _env("ARC_TELEMETRY", "/kaggle/working/telemetry.jsonl", str),
    min_prob         = _env("ARC_MIN_PROB", 0.2, float),
    extra_min_prob   = _env("ARC_EXTRA_MIN_PROB", 0.12, float),
    window_per_view  = _env("ARC_WINDOW_PER_VIEW", 135.0, float),   # baseline: 540 s per 4-view batch
    max_rows         = _env("ARC_MAX_ROWS", 8, int),
    rows8_max_ctx    = _env("ARC_ROWS8_MAX_CTX", 3600, int),
    merge_lora       = _env("ARC_MERGE_LORA", 0, int),   # off: bf16 rounding of W + B@A erodes the TTT update
    bench            = _env("ARC_BENCH", 0, int),
)

class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss

class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch

# Minimal performance patch: preserve the baseline beam set and ranking, but transfer
# only the 12 ARC-token NLL values to CPU instead of every Qwen vocabulary logit.
_ARC_TOKEN_ID_CACHE = {}

def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids

DFS_STEPS = [0]

def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, window) -> dict:

    n = logits.size(0)

    # Algebraically identical to: scores - logits.float().cpu().log_softmax(-1),
    # restricted to the same ARC_TOKENS used by the baseline DFS loop.
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])

    while time.time() - start_time < window and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        DFS_STEPS[0] += 1
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            window=window,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes

@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, window):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        window=window,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result

@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    # Keep logits on GPU and gather only the target-token scores. KV cache is not
    # consumed by teacher-forced scoring, so disabling it removes redundant writes.
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result

def stable_seed(key, offset=0):
    # The baseline used hash(bk), which is salted per process (PYTHONHASHSEED unset);
    # crc32 keeps the 8 rescoring views identical across workers and reruns.
    return (zlib.crc32(key.encode("utf-8")) + offset) % (1024 ** 2)

def resolve_model_dir():
    if CFG["model_path"] and os.path.isfile(os.path.join(CFG["model_path"], "config.json")):
        return CFG["model_path"]
    candidates = [\
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",\
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "grids15" in d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("qwen3_4b_grids15_sft139 model directory not found under /kaggle/input")

# ---------------------------------------------------------------------------
# Inference-time LoRA merge.
# Unsloth's fast inference path applies the r=256 adapter as two extra matmuls per
# projection (14 per layer). After TTT we fold B@A into the frozen base weights and mark
# the layer merged (unsloth's get_lora_parameters and PEFT both skip the adapter then).
# The original bf16 base weights are restored bit-exactly from a CPU snapshot before
# the next task trains, so no rounding drift accumulates across tasks.
# ---------------------------------------------------------------------------
class LoraMerger:

    def __init__(self, model, snapshot_path=None):
        from peft.tuners.lora import LoraLayer
        self.layers = []
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and isinstance(getattr(module, "base_layer", None), torch.nn.Linear):
                if "default" in module.lora_A and isinstance(getattr(module, "merged_adapters", None), list):
                    self.layers.append((name, module))
        if snapshot_path is None:
            self.snapshot = {
                name: module.base_layer.weight.detach().to("cpu", copy=True)
                for name, module in self.layers
            }
        else:
            self.snapshot = self._shared_snapshot(snapshot_path)
            # the file was written by whichever worker got there first: check it is ours
            for name, module in self.layers[:: max(1, len(self.layers) // 8)]:
                if not torch.equal(module.base_layer.weight.detach().cpu(), self.snapshot[name]):
                    raise RuntimeError(f"base-weight snapshot mismatch at {name}")
        self.merged = False

    def _shared_snapshot(self, path):
        """One copy of the frozen base weights for all workers: written once to local disk,
        memory-mapped by every worker so the page cache is shared (4 private copies would
        cost ~29 GB of host RAM)."""
        lock = path + ".lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owner = True
        except FileExistsError:
            owner = False
        if owner:
            state = {name: module.base_layer.weight.detach().to("cpu", copy=True) for name, module in self.layers}
            torch.save(state, path + ".tmp")
            del state
            os.replace(path + ".tmp", path)
        else:
            t0 = time.time()
            while not os.path.exists(path):
                if time.time() - t0 > 300:
                    raise TimeoutError("shared base-weight snapshot was never written")
                time.sleep(2)
        return torch.load(path, mmap=True, weights_only=True)

    @torch.no_grad()
    def merge(self):
        self.merged = True  # set first: restore() must run even if merging stops half-way
        for name, module in self.layers:
            W = module.base_layer.weight
            A = module.lora_A["default"].weight.float()
            B = module.lora_B["default"].weight.float()
            s = module.scaling["default"]
            W.copy_((W.float() + (B @ A) * s).to(W.dtype))
            module.merged_adapters.append("default")

    @torch.no_grad()
    def restore(self):
        if not self.merged:
            return
        for name, module in self.layers:
            module.base_layer.weight.copy_(self.snapshot[name], non_blocking=False)
            module.merged_adapters.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.merged = False

def view_geometry(subkey):
    ops = subkey.split(".")[1:]
    transposed = len(ops) > 0 and ops[0] == "transpose"
    rot = sum(1 for op in ops if op == "rot90")
    return transposed, rot

def make_batches(eval_ds, formatter, tokenizer, max_rows, rows8_max_ctx):
    """Group decoded views into DFS batches.

    Turbo DFS shares one position id across rows, so every row of a batch must have the
    same prompt length; views are grouped by their exact token count (shape class
    {id, rot180, T.rot90, T.rot270} vs {rot90, rot270, T, T.rot180}, minus any
    examples cut_to_len dropped). Short prompts decode 8 rows per batch, long ones 4.
    Batches are interleaved across test outputs so a time cap leaves every output
    with views from both shape classes, as in the baseline order."""
    per_test = defaultdict(lambda: defaultdict(list))
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        tokens = tokenizer.encode(eval_ds.get(subkey, formatter)["input"])
        per_test[test_id][len(tokens)].append((subkey, tokens))
    chunks_per_test = {}
    for test_id, groups in per_test.items():
        chunks = []
        # largest group first == the class containing the identity view in normal cases
        for length, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rows = max_rows if length <= rows8_max_ctx else min(4, max_rows)
            for i in range(0, len(items), rows):
                chunks.append(items[i:i + rows])
        # alternate shape classes: A1, B1, A2, B2 ...
        half = defaultdict(list)
        for c in chunks:
            half[len(c[0][1])].append(c)
        ordered = []
        lists = list(half.values())
        for i in range(max(len(l) for l in lists)):
            for l in lists:
                if i < len(l):
                    ordered.append(l[i])
        chunks_per_test[test_id] = ordered
    batches = []
    n = max(len(c) for c in chunks_per_test.values())
    for i in range(n):
        for test_id in sorted(chunks_per_test):
            if i < len(chunks_per_test[test_id]):
                batches.append(chunks_per_test[test_id][i])
    return batches

def output_confident(results_by_bk, bk, n_views):
    """True if one candidate clearly dominates the views decoded so far."""
    counts = defaultdict(int)
    for sol in results_by_bk.get(bk, []):
        counts[sol] += 1
    if not counts:
        return False
    top = sorted(counts.values(), reverse=True)
    c1 = top[0]
    c2 = top[1] if len(top) > 1 else 0
    return c1 >= max(6, n_views // 2) and c1 >= 2 * c2 + 2

def tel(path, rec):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

def worker(rank, end_time, shared, n_workers, test_path):

    # turbo_dfs recurses once per generated token (up to 931 for a 30x30 grid)
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    bench_pending = bool(CFG["bench"])

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolve_model_dir(),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    merger = None
    if CFG["merge_lora"]:
        try:
            merger = LoraMerger(model, snapshot_path=os.getenv("ARC_SNAPSHOT", "/tmp/arc_base_weights.pt"))
            print(f"[Rank {rank}] LoRA merge enabled over {len(merger.layers)} layers")
            if not merger.layers:
                merger = None
        except Exception as e:
            print(f"[Rank {rank}] LoRA merge disabled: {e!r}")
            merger = None
    merge_checked = False

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = CFG["out_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    pacer = Pacer(shared, rank, n_workers, end_time, arc_test_set.queries)

    def bench_dfs(items, label):
        tokens = [t for _, t in items]
        torch.cuda.synchronize()
        t0 = time.time()
        steps0 = DFS_STEPS[0]
        with torch.inference_mode():
            res = inference_turbo_dfs(model, tokens, max_new_tokens, -np.log(CFG["min_prob"]), end_time, 600.0)
        torch.cuda.synchronize()
        dt = time.time() - t0
        steps = DFS_STEPS[0] - steps0
        beams = sorted((b, tuple(tk)) for _, bs in res for b, tk in [(round(x[0], 3), x[1]) for x in bs])
        print(f"[Rank {rank}] BENCH {label}: rows {len(items)} ctx {len(tokens[0])} time {dt:.1f}s "
              f"steps {steps} ms/step {1000 * dt / max(steps, 1):.1f} beams {len(beams)}")
        return dict(label=label, rows=len(items), ctx=len(tokens[0]), t=round(dt, 2), steps=steps), beams

    def run_task(key, plan):
        nonlocal model, merge_checked, merger, bench_pending

        start_time = time.time()
        deadline = min(start_time + plan["budget"], end_time)
        info = dict(task=key, rank=rank, pass_name=plan["pass_name"], level=plan["level"],
                    budget=round(plan["budget"], 1), est=round(plan["est"], 1), kappa=round(plan["kappa"], 3))

        torch.cuda.reset_peak_memory_stats()

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=plan["n_train_aug"], shfl_keys=True, seed=plan["train_seed"])
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        train_args_task = dict(train_args, seed=plan["lora_seed"])
        if plan["lora_seed"] != 42:
            torch.manual_seed(plan["lora_seed"])

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args_task),
            )

            stats = trainer.train()

            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

            del trainer

        model = FastLanguageModel.for_inference(model)

        gc.collect()
        torch.cuda.empty_cache()

        info["train_s"] = round(time.time() - start_time, 1)
        info["train_steps"] = int(stats.global_step)
        info["train_loss"] = round(float(stats.training_loss), 5)
        info["train_mem_mb"] = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] trained {key} ({plan['pass_name']}, level {plan['level']}, "
              f"{stats.global_step} steps) in {info['train_s']}s, loss {info['train_loss']}")

        torch.cuda.reset_peak_memory_stats()

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        eval_ds = puzzle_ds_multi.augment(n=2, seed=plan["eval_seed"])
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        batches = make_batches(eval_ds, formatter, tokenizer, CFG["max_rows"], CFG["rows8_max_ctx"])

        bench = []
        if bench_pending and merger is not None and len(batches[0]) >= 8:
            bench_pending = False
            r, beams_unmerged = bench_dfs(batches[0], "unmerged")
            bench.append(r)

        # --- merge the adapter for decoding (verified once per worker) ---
        if merger is not None:
            try:
                probe_q = [eval_ds.get(batches[0][0][0], formatter)["input"]]
                probe_a = [formatter.fmt_reply([puzzle_ds_multi.queries[batches[0][0][0].split(".")[0]]["test"][0]["input"]])]
                if not merge_checked:
                    with torch.inference_mode():
                        ref = calc_scores(probe_q, probe_a, tokenizer, model)[0]
                merger.merge()
                if not merge_checked:
                    with torch.inference_mode():
                        got = calc_scores(probe_q, probe_a, tokenizer, model)[0]
                    ok = abs(got - ref) <= 0.03 * abs(ref) + 1.0
                    print(f"[Rank {rank}] merge self-test: unmerged NLL {ref:.4f} merged NLL {got:.4f} -> {'OK' if ok else 'MISMATCH'}")
                    info["merge_check"] = [round(ref, 4), round(got, 4), ok]
                    merge_checked = True
                    if not ok:
                        merger.restore()
                        merger = None
            except Exception as e:
                print(f"[Rank {rank}] merge failed, continuing unmerged: {e!r}")
                try:
                    merger.restore()
                except Exception:
                    pass
                merger = None

        if bench and merger is not None and merger.merged:
            r, beams_merged = bench_dfs(batches[0], "merged")
            bench.append(r)
            r["same_beams"] = [t for _, t in beams_merged] == [t for _, t in beams_unmerged]
            r, _ = bench_dfs(batches[0][:4], "merged-4rows")
            bench.append(r)
            info["bench"] = bench

        known_scores = {}
        sols_by_bk = defaultdict(list)
        n_views_done = defaultdict(int)
        dfs_log = []

        def decode_batch(items, max_score, suffix):
            subkeys = [s for s, _ in items]
            tokens = [t for _, t in items]
            window = CFG["window_per_view"] * len(items)
            t0 = time.time()
            steps0 = DFS_STEPS[0]
            try:
                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, min(deadline, end_time), window)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if len(items) == 1:
                    print(f"[Rank {rank}] OOM on single view {subkeys[0]}, skipped")
                    return
                print(f"[Rank {rank}] OOM with {len(items)} rows, splitting")
                h = len(items) // 2
                decode_batch(items[:h], max_score, suffix)
                decode_batch(items[h:], max_score, suffix)
                return
            dt = time.time() - t0
            steps = DFS_STEPS[0] - steps0
            dfs_log.append((len(items), len(tokens[0]), round(dt, 1), steps, round(1000 * dt / max(steps, 1), 1)))
            for subkey in subkeys:
                n_views_done[subkey.split(".")[0]] += 1

            for subkey_id, scored_beams in dfs_result:

                subkey = subkeys[subkey_id]
                bk = subkey.split(".")[0]
                decoded_result = []

                for beam_score, toks in scored_beams:

                    array = formatter.convert_tokens_to_array(toks)
                    if array is None:
                        continue

                    solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                    grid_id = (bk, tuple(map(tuple, solution)))

                    if grid_id in known_scores:
                        augmented_scores = known_scores[grid_id]
                    else:
                        aug_dataset = ArcDataset(
                            keys=[bk],
                            queries={bk: puzzle_ds_multi.queries.get(bk)},
                            replies={bk: [solution.tolist()]},
                        )
                        aug_dataset = aug_dataset.augment(seed=stable_seed(bk))
                        aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                        aug_queries = []
                        aug_answers = []
                        for augmented_sample in aug_dataset.as_list(formatter):
                            aug_queries.append(augmented_sample["input"])
                            aug_answers.append(augmented_sample["reply"])
                        augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                        augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                        augmented_scores = augmented_scores1 + augmented_scores2
                        known_scores[grid_id] = augmented_scores

                    decoded_result.append({
                        "beam_score": beam_score,
                        "score_aug": augmented_scores,
                        "solution": solution,
                    })
                    sols_by_bk[bk].append(grid_id[1])

                if len(decoded_result):
                    with bz2.BZ2File(os.path.join(dir_outputs, subkey + suffix), "w") as f:
                        pickle.dump(decoded_result, f)

        timed_out = False
        with torch.inference_mode():

            for items in batches:
                if time.time() > deadline - 5 or time.time() > end_time - 30:
                    timed_out = True
                    break
                decode_batch(items, -np.log(CFG["min_prob"]), plan["suffix"])

            info["base_s"] = round(time.time() - start_time, 1)

            # --- adaptive extra views for outputs without a dominant candidate ---
            extra_done = 0
            if plan["extra_views"] and not timed_out:
                bks = sorted(puzzle_ds_multi.keys)
                todo = [bk for bk in bks if not output_confident(sols_by_bk, bk, n_views_done[bk])]
                if todo:
                    extra_ds = puzzle_ds_multi.change_keys(todo).augment(n=1, seed=plan["eval_seed"] + 1000)
                    extra_ds = extra_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                    extra_batches = make_batches(extra_ds, formatter, tokenizer, CFG["max_rows"], CFG["rows8_max_ctx"])
                    for items in extra_batches:
                        if time.time() > deadline - 5 or time.time() > end_time - 30:
                            break
                        decode_batch(items, -np.log(CFG["extra_min_prob"]), plan["suffix"] + ".x")
                        extra_done += len(items)
                info["extra_outputs"] = len(todo)
            info["extra_views"] = extra_done

        timed_out = timed_out or time.time() >= deadline - 2

        if merger is not None:
            merger.restore()

        # uncertainty summary for the tail pass
        unc = {}
        for bk in sorted(puzzle_ds_multi.keys):
            counts = defaultdict(int)
            for s in sols_by_bk.get(bk, []):
                counts[s] += 1
            top = sorted(counts.values(), reverse=True)
            unc[bk] = dict(n_unique=len(counts), c1=top[0] if top else 0,
                           c2=top[1] if len(top) > 1 else 0, views=n_views_done[bk])
        info["unc"] = unc
        info["dfs"] = dfs_log
        info["infer_mem_mb"] = torch.cuda.max_memory_allocated() // 1024**2
        info["elapsed"] = round(time.time() - start_time, 1)
        info["timed_out"] = timed_out
        tel(CFG["telemetry"], info)
        print(f"[Rank {rank}] finished {key} ({plan['pass_name']}) in {info['elapsed']}s "
              f"[train {info['train_s']}s, base views {info['base_s']}s, extra views {extra_done}, "\
              f"budget {plan['budget']:.0f}s, dfs {dfs_log}]")
        return info

    while True:
        plan = pacer.next_task()
        if plan is None:
            break
        key = plan["task"]
        t0 = time.time()
        try:
            info = run_task(key, plan)
            pacer.done(plan, time.time() - t0, info)
        except Exception as e:
            print(f"[Rank {rank}] ERROR on {key}: {e!r}")
            traceback.print_exc()
            pacer.done(plan, time.time() - t0, None)
            try:
                if merger is not None:
                    merger.restore()
            except Exception as e2:
                print(f"[Rank {rank}] restore after error failed: {e2!r}; disabling merge")
                merger = None
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Rank {rank}] no more work")
```

```
Writing arc_solver.py
```

In \[7\]:

linkcode

```
%%writefile starter.py
import os
import time
import json
import traceback
import torch
import argparse
import torch.multiprocessing as mp

N_WORKERS = int(os.getenv("ARC_N_WORKERS", "4"))
FLAG_DIR = os.getenv("ARC_FLAG_DIR", "/kaggle")

def local_worker(rank, end_time, shared, test_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"{FLAG_DIR}/worker{rank-1}"):
            time.sleep(5)

    # A raised exception would make mp.spawn terminate the other three workers.
    worker = None
    try:
        from arc_solver import worker
    except BaseException as e:
        print(f"[Rank {rank}] FATAL import: {e!r}")
        traceback.print_exc()
    finally:
        with open(f"{FLAG_DIR}/worker{rank}", "w") as f:
            f.write("Ok")

    print(f"[Rank {rank}] start!")

    if worker is not None:
        try:
            worker(rank, end_time, shared, N_WORKERS, test_path)
        except BaseException as e:
            print(f"[Rank {rank}] FATAL: {e!r}")
            traceback.print_exc()

    print(f"[Rank {rank}] done!")

def resolve_test_path(rerun_mode):
    if os.getenv("ARC_TEST_PATH"):
        return os.getenv("ARC_TEST_PATH")
    name = "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json"
    for root in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return os.path.join("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", name)

def main(worker_fn=local_worker):

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    from arc_sched import est_task_seconds, est_train_seconds, est_decode_seconds

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    test_path = resolve_test_path(rerun_mode)

    with open(test_path, "r") as f:
        data = json.load(f)

    if not rerun_mode:
        keys = os.getenv("ARC_COMMIT_TASKS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
        data = {k: data[k] for k in keys if k in data}

    est = {k: est_task_seconds(t) for k, t in data.items()}
    # cheapest (and most often solved) tasks first: whatever the time pressure on the
    # hidden set turns out to be, it can only squeeze the large low-yield tail
    order = sorted(data.keys(), key=lambda k: (est[k], k))

    budget = args.end_time - time.time()
    print(f"[starter] {len(order)} tasks, predicted baseline work {sum(est.values())/3600:.2f} GPU-h, "
          f"available {N_WORKERS * budget / 3600:.2f} GPU-h, test_path={test_path}")

    manager = mp.Manager()
    shared = dict(
        lock=manager.Lock(),
        order=manager.list(order),
        est=manager.dict(est),
        state=manager.dict(qi=0, tr_act=0.0, tr_exp=0.0, dec_act=0.0, dec_exp=0.0,
                           rem_tr0=float(sum(est_train_seconds(t) for t in data.values())),
                           rem_dec0=float(sum(est_decode_seconds(t) for t in data.values()))),
        busy=manager.dict(),
        unc=manager.dict(),
        tail_done=manager.dict(),
        failed=manager.dict(),
    )

    mp.spawn(worker_fn, args=(args.end_time, shared, test_path), nprocs=N_WORKERS, join=True)

    st = dict(shared["state"])
    print(f"[starter] finished: main queue reached {st['qi']}/{len(order)}, tail runs {len(shared['tail_done'])}, "
          f"kappa train {(1500 + st['tr_act']) / (1500 + st['tr_exp']):.3f} decode {(1500 + st['dec_act']) / (1500 + st['dec_exp']):.3f}")

if __name__ == "__main__":
    main()
```

```
Writing starter.py
```

In \[8\]:

linkcode

```
import subprocess, sys
env = dict(
    os.environ,
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_BENCH="0",
    ARC_COMMIT_TASKS="0934a4d8,36a08778,981571dc,aa4ec2a5",
)
rc = subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}"], env=env)
print("starter exit code:", rc, "| elapsed (h):", round((time.time() - T0) / 3600, 2))
```

```
[starter] 4 tasks, predicted baseline work 1.06 GPU-h, available 2.05 GPU-h, test_path=/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[Rank 0] start!
==((====))==  Unsloth 2025.9.7: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 8.9. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.32.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
```

```
Loading checkpoint shards:  50%|█████     | 1/2 [01:01<01:01, 61.82s/it]
```

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[Rank 1] start!
==((====))==  Unsloth 2025.9.7: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 8.9. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.32.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [01:26<00:00, 43.50s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:22<00:00, 11.32s/it]
```

```
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[Rank 2] start!
==((====))==  Unsloth 2025.9.7: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 8.9. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.32.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:01<00:00,  1.06it/s]
```

```
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[Rank 3] start!
==((====))==  Unsloth 2025.9.7: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 8.9. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.32.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:01<00:00,  1.06it/s]
```

```
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 0] trained 0934a4d8 (A, level 0, 128 steps) in 413.4s, loss 0.00348
[Rank 0] finished 0934a4d8 (A) in 580.9s [train 413.4s, base views 580.9s, extra views 0, budget 1603s, dfs [(4, 4793, 10.5, 65, 161.2), (4, 4793, 11.8, 78, 150.9), (4, 4793, 11.3, 73, 155.3), (4, 4793, 11.6, 78, 149.2)]]
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 1] trained aa4ec2a5 (A, level 0, 128 steps) in 320.5s, loss 0.00575
[Rank 1] finished aa4ec2a5 (A) in 676.6s [train 320.5s, base views 676.6s, extra views 0, budget 1603s, dfs [(4, 3917, 94.4, 960, 98.3), (4, 3932, 71.8, 727, 98.7), (4, 3917, 93.6, 958, 97.7), (4, 3932, 71.4, 727, 98.2)]]
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 2] trained 36a08778 (A, level 0, 128 steps) in 202.0s, loss 0.00511
[Rank 2] finished 36a08778 (A) in 823.3s [train 202.0s, base views 823.3s, extra views 0, budget 1563s, dfs [(8, 2356, 30.6, 271, 113.1), (8, 3014, 227.7, 2327, 97.8), (8, 2360, 58.6, 581, 100.9), (8, 3018, 214.5, 2218, 96.7)]]
[Rank 2] no more work
[Rank 2] done!
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 0] trained 0934a4d8 (A, level 0, 128 steps) in 413.4s, loss 0.00348
[Rank 0] finished 0934a4d8 (A) in 580.9s [train 413.4s, base views 580.9s, extra views 0, budget 1603s, dfs [(4, 4793, 10.5, 65, 161.2), (4, 4793, 11.8, 78, 150.9), (4, 4793, 11.3, 73, 155.3), (4, 4793, 11.6, 78, 149.2)]]
[Rank 0] trained 0934a4d8 (B, level 0, 128 steps) in 402.6s, loss 0.00453
[Rank 0] finished 0934a4d8 (B) in 564.1s [train 402.6s, base views 564.1s, extra views 0, budget 1012s, dfs [(4, 4793, 9.3, 53, 176.1), (4, 4793, 9.1, 50, 182.6), (4, 4793, 9.2, 51, 180.8), (4, 4793, 11.6, 75, 154.3)]]
[Rank 0] no more work
[Rank 0] done!
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 3] trained 981571dc (A, level 0, 128 steps) in 829.9s, loss 0.00021
[Rank 3] finished 981571dc (A) in 1215.9s [train 829.9s, base views 1215.9s, extra views 0, budget 1502s, dfs [(4, 6534, 92.6, 929, 99.6), (4, 6534, 92.6, 929, 99.7), (4, 6534, 93.7, 929, 100.8), (4, 6534, 92.5, 929, 99.6)]]
[Rank 3] no more work
[Rank 3] done!
Unsloth: Training embed_tokens in mixed precision to save VRAM
Unsloth: Training lm_head in mixed precision to save VRAM
[Rank 1] trained aa4ec2a5 (A, level 0, 128 steps) in 320.5s, loss 0.00575
[Rank 1] finished aa4ec2a5 (A) in 676.6s [train 320.5s, base views 676.6s, extra views 0, budget 1603s, dfs [(4, 3917, 94.4, 960, 98.3), (4, 3932, 71.8, 727, 98.7), (4, 3917, 93.6, 958, 97.7), (4, 3932, 71.4, 727, 98.2)]]
[Rank 1] trained aa4ec2a5 (B, level 0, 128 steps) in 308.3s, loss 0.00567
[Rank 1] finished aa4ec2a5 (B) in 653.3s [train 308.3s, base views 653.3s, extra views 0, budget 916s, dfs [(4, 3917, 71.8, 728, 98.7), (4, 3932, 71.5, 727, 98.4), (4, 3917, 72.0, 728, 98.9), (4, 3932, 104.4, 1058, 98.6)]]
[Rank 1] no more work
[Rank 1] done!
[starter] finished: main queue reached 4/4, tail runs 2, kappa train 0.970 decode 0.842
starter exit code: 0 | elapsed (h): 0.45
```

In \[9\]:

linkcode

```
import os, json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

data_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
if not os.path.isdir(data_dir):
    data_dir = "/kaggle/input/arc-prize-2026-arc-agi-2"

if RERUN:
    data = ArcDataset.from_file(f"{data_dir}/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file(f"{data_dir}/arc-agi_evaluation_challenges.json")
    data = data.load_replies(f"{data_dir}/arc-agi_evaluation_solutions.json")

try:
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    if os.path.isdir("/kaggle/inference_outputs"):
        decoder.load_decoded_results("/kaggle/inference_outputs")
    submission = data.get_submission(decoder.run_selection_algo())
except Exception as e:
    print("*** selection failed, writing placeholder submission:", repr(e))
    submission = data.get_submission()

with open("submission.json", "w") as f:
    json.dump(submission, f)
print("tasks in submission:", len(submission))

if not RERUN:
    try:
        decoder.benchmark_selection_algos()
    except Exception as e:
        print("benchmark failed:", repr(e))
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))
```

```
*** Load solutions from '/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json'...
*** Generating submission for 5 outputs...
tasks in submission: 120
*** Benchmark selection algorithms...
ALL_CORRECT: 0.00938 -  0.00525 30x30 [981571dc_0.rot90.rot90.permute8579032416.ex201.out0]
ALL_CORRECT: 0.01521 -  0.00525 30x30 [981571dc_0.rot90.rot90.permute5386701942.ex213.out0]
ALL_CORRECT: 0.00158 -  0.00525 30x30 [981571dc_0.transpose.rot90.permute4907281356.ex210.out0]
ALL_CORRECT: 0.00101 -  0.00525 30x30 [981571dc_0.permute9302675184.ex302.out0]
ALL_CORRECT: 0.00096 -  0.00525 30x30 [981571dc_0.rot90.permute4950812673.ex013.out0]
ALL_CORRECT: 0.00495 -  0.00525 30x30 [981571dc_0.transpose.rot90.rot90.rot90.permute7690351428.ex231.out0]
ALL_CORRECT: 0.00133 -  0.00525 30x30 [981571dc_0.transpose.permute1602397845.ex013.out0]
ALL_CORRECT: 0.00306 -  0.00525 30x30 [981571dc_0.rot90.rot90.rot90.permute1654278039.ex120.out0]
ALL_CORRECT: 0.00748 -  0.00525 30x30 [981571dc_0.transpose.rot90.rot90.permute5436201789.ex301.out0]
ALL_CORRECT: 0.00089 -  0.00525 30x30 [981571dc_0.rot90.permute4281673590.ex213.out0]
ALL_CORRECT: 0.00176 -  0.00525 30x30 [981571dc_0.transpose.permute9180254367.ex230.out0]
ALL_CORRECT: 0.00613 -  0.00525 30x30 [981571dc_0.transpose.rot90.rot90.permute0986123745.ex103.out0]
ALL_CORRECT: 0.00185 -  0.00525 30x30 [981571dc_0.rot90.rot90.rot90.permute4965803712.ex023.out0]
ALL_CORRECT: 0.00902 -  0.00525 30x30 [981571dc_0.transpose.rot90.rot90.rot90.permute6142358709.ex120.out0]
ALL_CORRECT: 0.00225 -  0.00525 30x30 [981571dc_0.permute4150723698.ex312.out0]
ALL_CORRECT: 0.00163 -  0.00525 30x30 [981571dc_0.transpose.rot90.permute5980213764.ex023.out0]
ALL_CORRECT: 0.00208 -  0.60987 26x27 [aa4ec2a5_0.rot90.rot90.rot90.permute2968075314.ex012.p2.out0]
ALL_CORRECT: 0.12725 -  0.60987 26x27 [aa4ec2a5_0.rot90.permute9158362407.ex021.p2.out0]
ALL_CORRECT: 0.00018 -  1.98582 26x27 [aa4ec2a5_0.rot90.permute4950812673.ex201.out0]
ALL_CORRECT: 0.01138 -  0.60987 26x27 [aa4ec2a5_0.transpose.rot90.rot90.permute0659473182.ex201.p2.out0]
ALL_CORRECT: 0.00431 -  0.60987 26x27 [aa4ec2a5_0.transpose.rot90.rot90.permute2076584391.ex012.p2.out0]
ALL_CORRECT: 0.01151 -  1.98582 26x27 [aa4ec2a5_0.transpose.rot90.rot90.rot90.permute6142358709.ex021.out0]
ALL_CORRECT: 0.01849 -  0.60987 26x27 [aa4ec2a5_0.rot90.permute7205186349.ex012.p2.out0]
ALL_CORRECT: 0.00108 -  0.60987 26x27 [aa4ec2a5_0.transpose.rot90.rot90.rot90.permute1659078432.ex210.p2.out0]
ALL_CORRECT: 0.00230 -  0.60987 26x27 [aa4ec2a5_0.permute8617029435.ex120.p2.out0]
ALL_CORRECT: 1.31389 -  0.60987 26x27 [aa4ec2a5_0.transpose.rot90.permute8095316724.ex102.p2.out1]
ALL_CORRECT: 0.00308 -  0.60987 26x27 [aa4ec2a5_0.rot90.rot90.rot90.permute9201758346.ex120.p2.out0]
ALL_CORRECT: 0.00119 -  1.98582 26x27 [aa4ec2a5_0.rot90.permute4281673590.ex102.out0]
ALL_CORRECT: 0.00142 -  0.60987 26x27 [aa4ec2a5_0.transpose.rot90.rot90.rot90.permute0876324195.ex201.p2.out0]
ALL_CORRECT: 0.00746 -  0.60987 26x27 [aa4ec2a5_0.permute7146980235.ex120.p2.out0]
ALL_CORRECT: 0.00121 -  1.98582 26x27 [aa4ec2a5_0.transpose.rot90.rot90.permute0986123745.ex021.out0]
ALL_CORRECT: 0.01360 -  1.98582 26x27 [aa4ec2a5_0.permute9302675184.ex012.out0]
ALL_CORRECT: 0.00311 -  1.98582 26x27 [aa4ec2a5_0.transpose.rot90.rot90.rot90.permute7690351428.ex210.out0]
ALL_CORRECT: 0.00062 -  1.98582 26x27 [aa4ec2a5_0.transpose.rot90.rot90.permute5436201789.ex021.out0]
ALL_CORRECT: 0.02037 -  1.98582 26x27 [aa4ec2a5_0.permute4150723698.ex012.out0]
ALL_CORRECT: 0.02682 -  3.95482 16x16 [36a08778_0.transpose.permute9316478025.ex301425.out0]
ALL_CORRECT: 0.00015 -  3.95482 16x16 [36a08778_0.transpose.rot90.rot90.rot90.permute4965803712.ex420135.out0]
ALL_CORRECT: 0.18222 -  3.95482 16x16 [36a08778_0.transpose.rot90.permute3170468295.ex120534.out0]
ALL_CORRECT: 0.00012 -  3.95482 16x16 [36a08778_0.transpose.rot90.rot90.rot90.permute7568409321.ex152304.out0]
ALL_CORRECT: 0.11082 -  3.95482 16x16 [36a08778_0.rot90.permute1258796043.ex532401.out0]
ALL_CORRECT: 0.00001 -  3.95482 16x16 [36a08778_0.permute6720984531.ex021543.out0]
ALL_CORRECT: 0.08098 -  3.95482 16x16 [36a08778_0.rot90.rot90.permute9302675184.ex105324.out0]
ALL_CORRECT: 0.21806 -  3.95482 16x16 [36a08778_0.transpose.rot90.permute1654278039.ex204513.out0]
ALL_CORRECT: 0.13700 -  3.95482 16x16 [36a08778_0.rot90.rot90.permute7931086254.ex405132.out0]
ALL_CORRECT: 0.01529 -  3.95482 16x16 [36a08778_0.rot90.permute5386701942.ex542013.out0]
ALL_CORRECT: 0.00787 -  3.95482 16x16 [36a08778_0.transpose.permute4950812673.ex132450.out0]
ALL_CORRECT: 0.00000 -  3.95482 16x16 [36a08778_0.permute4150723698.ex031524.out0]
 subkeys: 47/127
 avg correct beam score:  0.05090
 max correct beam score:  1.31389
{'36a08778_0', '981571dc_0', 'aa4ec2a5_0'}
 acc:   2.5/  4 ('score_full_probmul_3')
{'36a08778_0', '981571dc_0', 'aa4ec2a5_0'}
 acc:   2.5/  4 ('score_kgmon')
*** Reload score: 2.5
```

In \[10\]:

linkcode

```
# Commit-run diagnostics: per-task timing and DFS speed.
import json, os
path = "/kaggle/working/telemetry.jsonl"
if os.path.exists(path):
    for line in open(path):
        r = json.loads(line)
        r.pop("unc", None)
        print(json.dumps(r))
```

```
{"task": "0934a4d8", "rank": 0, "pass_name": "A", "level": 0, "budget": 1603.1, "est": 610.0, "kappa": 1.0, "train_s": 413.4, "train_steps": 128, "train_loss": 0.00348, "train_mem_mb": 13011, "base_s": 580.9, "extra_views": 0, "dfs": [[4, 4793, 10.5, 65, 161.2], [4, 4793, 11.8, 78, 150.9], [4, 4793, 11.3, 73, 155.3], [4, 4793, 11.6, 78, 149.2]], "infer_mem_mb": 14719, "elapsed": 580.9, "timed_out": false}
{"task": "aa4ec2a5", "rank": 1, "pass_name": "A", "level": 0, "budget": 1603.1, "est": 862.3, "kappa": 1.0, "train_s": 320.5, "train_steps": 128, "train_loss": 0.00575, "train_mem_mb": 13011, "base_s": 676.6, "extra_views": 0, "dfs": [[4, 3917, 94.4, 960, 98.3], [4, 3932, 71.8, 727, 98.7], [4, 3917, 93.6, 958, 97.7], [4, 3932, 71.4, 727, 98.2]], "infer_mem_mb": 14102, "elapsed": 676.6, "timed_out": false}
{"task": "36a08778", "rank": 2, "pass_name": "A", "level": 0, "budget": 1563.4, "est": 1009.3, "kappa": 1.0, "train_s": 202.0, "train_steps": 128, "train_loss": 0.00511, "train_mem_mb": 13011, "base_s": 823.3, "extra_views": 0, "dfs": [[8, 2356, 30.6, 271, 113.1], [8, 3014, 227.7, 2327, 97.8], [8, 2360, 58.6, 581, 100.9], [8, 3018, 214.5, 2218, 96.7]], "infer_mem_mb": 17128, "elapsed": 823.3, "timed_out": false}
{"task": "0934a4d8", "rank": 0, "pass_name": "B", "level": 0, "budget": 1012.1, "est": 610.0, "kappa": 0.982, "train_s": 402.6, "train_steps": 128, "train_loss": 0.00453, "train_mem_mb": 13012, "base_s": 564.1, "extra_views": 0, "dfs": [[4, 4793, 9.3, 53, 176.1], [4, 4793, 9.1, 50, 182.6], [4, 4793, 9.2, 51, 180.8], [4, 4793, 11.6, 75, 154.3]], "infer_mem_mb": 14719, "elapsed": 564.1, "timed_out": false}
{"task": "981571dc", "rank": 3, "pass_name": "A", "level": 0, "budget": 1501.9, "est": 1340.9, "kappa": 1.0, "train_s": 829.9, "train_steps": 128, "train_loss": 0.00021, "train_mem_mb": 13150, "base_s": 1215.9, "extra_views": 0, "dfs": [[4, 6534, 92.6, 929, 99.6], [4, 6534, 92.6, 929, 99.7], [4, 6534, 93.7, 929, 100.8], [4, 6534, 92.5, 929, 99.6]], "infer_mem_mb": 17109, "elapsed": 1215.9, "timed_out": false}
{"task": "aa4ec2a5", "rank": 1, "pass_name": "B", "level": 0, "budget": 916.5, "est": 862.3, "kappa": 0.91, "train_s": 308.3, "train_steps": 128, "train_loss": 0.00567, "train_mem_mb": 13012, "base_s": 653.3, "extra_views": 0, "dfs": [[4, 3917, 71.8, 728, 98.7], [4, 3932, 71.5, 727, 98.4], [4, 3917, 72.0, 728, 98.9], [4, 3932, 104.4, 1058, 98.6]], "infer_mem_mb": 14102, "elapsed": 653.3, "timed_out": false}
```

## License

This Notebook has been released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.

## Continue exploring

- input







Input

3 files

- output







Output

47 files

- list\_alt







Logs

1700.1 second run - successful

- comment







Comments

0 comments


reCAPTCHA

Recaptcha requires verification.

protected by **reCAPTCHA**