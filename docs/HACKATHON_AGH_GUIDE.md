# Jet-vasc AGH deployment and operation guide

## 1. What AGH does in this project

Agnes Harness is the agent runtime and evidence layer. It plans the task, checks prerequisites, asks for approval before GPU work, invokes the Vasc-Gaussian tool plugin, reads execution feedback, verifies artifacts, classifies failures, and compares runs. Vasc-Gaussian remains the scientific reconstruction engine.

The connection is deliberately narrow: AGH can choose only a committed profile and `quick` or `full` mode. It cannot send arbitrary shell commands or arbitrary filesystem paths through this plugin.

## 2. Server prerequisites

- The latest Vasc-Gaussian repository checkout and its working CUDA/Python environment.
- Node.js 24.10 or newer.
- pnpm 10.34.5.
- Agnes Harness built from its official repository.
- The Agnes model key entered in AGH Web settings only.

Install the competition-only Python additions inside the existing `x_gaussian` environment:

```bash
python -m pip install -r requirements-competition.txt
```

## 3. Build AGH

From a directory beside Vasc-Gaussian:

```bash
git clone https://github.com/AgnesAI-Labs/agnes-harness.git
cd agnes-harness
corepack enable
corepack prepare pnpm@10.34.5 --activate
pnpm install --frozen-lockfile
pnpm --filter @agnes/cli build:local
```

## 4. Start AGH with Vasc-Gaussian attached

Use the actual paths printed by `pwd` and `which python`:

```bash
export VASC_GAUSSIAN_ROOT=/root/autodl-tmp/Vasc-Gaussian-sync
export VASC_PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/Vasc-Gaussian-sync
node /root/autodl-tmp/agnes-harness/packages/cli/dist/local/agnes.mjs serve
```

After the first successful build, the repository helper can start the same service while deriving the Vasc-Gaussian path automatically:

```bash
conda activate x_gaussian
cd /root/autodl-tmp/Vasc-Gaussian-sync
bash agh/start_agh.sh /root/autodl-tmp/agnes-harness
```

Do not put the Agnes model key in these environment variables or in Git. Open the address printed by AGH, select the Agnes AI provider in Settings, and paste the newly rotated key into the password field.

## 5. Install and enable the plugin

AGH accepts a local `file:` source only when it is a contained path below the selected workspace. Because the service was started from Vasc-Gaussian, open Settings -> Plugin management -> Install from source and inspect/install:

```text
file:./agh/vasc-tools
```

Review the package name `@jet-vasc/agh-vasc-tools`, content hash, and capabilities, then enable it. The plugin runs local code with the server user's privileges; this review is a required trust boundary.

Alternatively, run the AGH CLI while the shell remains in the Vasc-Gaussian workspace:

```bash
cd /root/autodl-tmp/Vasc-Gaussian-sync
node /root/autodl-tmp/agnes-harness/packages/cli/dist/local/agnes.mjs package inspect file:./agh/vasc-tools
node /root/autodl-tmp/agnes-harness/packages/cli/dist/local/agnes.mjs install file:./agh/vasc-tools
```

Then use the integrity and capability hash printed by the inspection/install flow to trust and enable the exact snapshot.

## 6. Install the workspace Skill

Start AGH with the Vasc-Gaussian repository selected as the workspace. In AGH resources/Skills, inspect and trust `.agh/skills/vasc-reconstruction/SKILL.md`. Confirm that the skill is enabled for the session.

## 7. First end-to-end test

Send this prompt to Agnes:

```text
请使用 vasc-reconstruction Skill：先列出实验配置，然后对 aneurism-v4 的 quick 模式、GPU 0 做预检。预检未通过时不要启动，只告诉我最小修复步骤。
```

After `ready: true`, ask:

```text
启动 aneurism-v4 quick 运行。请先说明它会写入输出并消耗 GPU，等待我在审批框确认；启动后返回 run id。
```

When it finishes:

```text
检查这个 run id 的状态并验证证据。分别汇报二维、三维体数据和拓扑指标，同时说明 quick 结果不能作为最终科学结论。
```

## 8. Full run and evidence

Only after the quick loop passes, repeat with `full`. Keep the whole `agh_runs/<run-id>/` directory. It contains the code revision, resolved parameters, stage commands, event ledger, log, state, exported volume, evaluation metrics, and verification result.

If DGR export encounters CUDA OOM, the bridge retries that export once with a smaller batch. Training OOM is not silently altered because changing scientific hyperparameters would invalidate comparability.

## 9. Common failures

| Code | Meaning | Action |
|---|---|---|
| `PREFLIGHT_FAILED` | Missing data, prior, module, GPU, or disk | Fix only the listed issue and rerun preflight |
| `RUN_ALREADY_ACTIVE` | Another GPU pipeline is active | Inspect it; wait or explicitly stop it |
| `CUDA_OUT_OF_MEMORY` | GPU memory exhausted | Export retries once; training requires a reviewed low-memory profile |
| `MISSING_FILE` | A declared input disappeared | Restore the exact file; do not substitute unrecorded data |
| `MISSING_DEPENDENCY` | Python environment is incomplete | Install from `requirements-competition.txt` in the same environment |
| `STAGE_FAILED` | A stage returned nonzero | Read the bounded log tail and stage name before changing anything |

## 10. Security checklist

- Revoke the model key that appeared in a screenshot and create a new one.
- Never commit keys, raw private data, AGH traces, or `agh_runs/`.
- Do not expose the AGH port publicly without access control.
- Use only Agnes for language-model calls during the competition.
- Confirm with organizers that local scientific reconstruction networks are treated as domain computation rather than prohibited third-party language models.
