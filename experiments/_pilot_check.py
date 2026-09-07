import json

with open("results/pilot_task_run.json", encoding="utf-8") as f:
    data = json.load(f)

for s in data:
    print("--- seed", s["seed"])
    for c, r in s["results"].items():
        if isinstance(r, dict):
            tr = "/".join(f'{t}:{r["transfer"][t]:.2f}' for t in ("t1", "t2", "t3"))
            print(f'  {c:22s} overall={r["overall"]:.3f} known={r["known_acc"]:.3f} '
                  f'[{tr}] rules={r["n_rules"]} rev={r["revisions"]}')
print("chance:", data[0]["chance"])