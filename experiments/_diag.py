import sys
sys.path.insert(0, ".")
from experiments.task_experiment import run_behavioral_config, ABLATIONS
from environment.attribute_tasks import AttributeTaskGenerator

gen = AttributeTaskGenerator(seed=1002, train_size=600, known_size=120,
                             t1_size=60, t2_size=60, t3_size=40)
sets = gen.generate()
test_sets = {"t1": sets.t1, "t2": sets.t2, "t3": sets.t3}

r = run_behavioral_config("null_bp", sets.train, test_sets, sets.known, seed=11002)
print("null_bp overall", r.overall, "known", r.known_acc, "rev", r.revisions, "rules", r.n_rules)
from collections import Counter
ops = Counter(ru["provenance"][-1] for ru in r.trace)
print("last op by rule:", ops)

# Show a few active rules
for ru in r.trace[:12]:
    print(ru["status"], ru["condition"], "->", ru["action"], ru["evidence"])