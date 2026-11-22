# Pre-registered model sets

All sets use the same probe and the same runner; only the model list, the output directory and the GPU list changed
between runs. The copies that each set carried on its run machine were byte-identical and are kept once here:

| Shared file | Used by | md5 |
|---|---|---|
| `../detection_gate/kvprobe_unified.py` | fitting set, PRED-P, N1, N4, N13 (rule features) | `09b2b0b5` |
| `predp/pp_rules.py` | every set (`CFGS`, thresholds, features, metrics) | `f8ca672f` |
| `predp/run_pp_probe.sh` | every set: set `X=$KVREL_WORK/<set>` and the model list | - |
| `n1/n1_rules.py` | N1 (blind, `SELECT_N = 9`) and N4 (injected; its `n4_rules.py` was identical) | `bc830dc9` |

| Set | Paper name | Rules | Models |
|---|---|---|---|
| `predp` | prospective | `pp_rules.py`, amendment `pp_amend_score.py` | `models_pp*.txt` |
| `n1` | blind | `n1_rules.py`; GGUF repairs of the blind candidates: `n1r_convert.py`, `n1r_repair.sh` | `models_n1*.txt` |
| `n4` | injected | `../n1/n1_rules.py` | `models_n4*.txt` |
| `n13` | new released | `n13_rules.py` (= `n1_rules.py` with `SELECT_N = 99`: every eligible candidate) | `models_n13*.txt` |

Before a run, copy the shared files and the set's own files into `$KVREL_WORK/<set>/code/` (the layout the runner
and the rules expect). Decisions are written by `<rules> decide` before any label exists; `score` reads the labels.
N13 was probed with a superset of `kvprobe_unified.py` that also computed a descriptive distortion feature; the
features the frozen rule reads come from the same code path as `kvprobe_unified.py`.
