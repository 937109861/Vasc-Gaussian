---
name: vasc-reconstruction
description: Plan, execute, verify, and compare Vasc-Gaussian sparse-view vessel reconstruction runs through the approved profile tools.
---

# Vasc-Gaussian reconstruction workflow

Use this Skill when the user asks to run, inspect, validate, troubleshoot, or compare a Vasc-Gaussian experiment.

1. Call `vasc_list_profiles` when the requested profile is not explicit.
2. Call `vasc_preflight` with the selected profile, mode, and GPU. Never start if `ready` is false. Explain each issue and suggest only the smallest corrective action.
3. Explain that `quick` verifies the execution loop but is not a scientific result; `full` runs the declared experiment.
4. Call `vasc_start_run` only after the user approves the AGH approval prompt. Never substitute shell commands or arbitrary paths.
5. Record the returned run id. Use `vasc_run_status` to inspect progress. Do not repeatedly poll without a user request.
6. If a stage fails, report its classified error. The bridge may retry DGR export once with a smaller batch after CUDA OOM. Do not silently change scientific hyperparameters.
7. When complete, call `vasc_verify_run`. Distinguish pipeline completeness from scientific or clinical validity.
8. Use `vasc_compare_runs` when a baseline and candidate run are available. Interpret PSNR/SSIM together with Dice, clDice, centerline recall, HD95, largest-component ratio, and small-component count.
9. Never describe this research system as a diagnostic medical device. Never expose model keys, local patient data, or raw private traces.

Every run must preserve `manifest.json`, `resolved_config.yaml`, `events.jsonl`, `run.log`, `status.json`, and metric outputs as evidence.
