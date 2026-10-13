# Agnes Harness integration

This directory exposes Vasc-Gaussian as allow-listed AGH tools rather than an unrestricted shell.

- `bridge.py`: profile validation, preflight, background execution, evidence, status, verification, comparison, and stopping.
- `profiles/`: committed experiment definitions. The agent can select a profile but cannot supply arbitrary commands.
- `vasc-tools/`: installable AGH backend plugin.
- `.agh/skills/vasc-reconstruction/SKILL.md`: the agent workflow and safety rules.

`quick` mode runs 100 iterations to verify the end-to-end loop. It must never be presented as the final scientific result. `full` preserves the repository v4 settings and runs the complete experiment.

Runtime outputs are written to `agh_runs/` and ignored by Git. The model key is configured only in the AGH Web settings; no key belongs in this repository.

## First-time server check

Run the read-only doctor before installing or starting anything:

```bash
bash agh/server_doctor.sh | tee agh_server_report.txt
```

The report hides key values. Send `agh_server_report.txt` to the setup assistant so missing Node, Python, CUDA, data, repository, or AGH components can be resolved in the right order.
