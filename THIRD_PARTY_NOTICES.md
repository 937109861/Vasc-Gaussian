# Third-party and pre-existing work disclosure

This repository contains pre-existing research code and third-party components. The AGH competition integration does not change the licenses or provenance of those components.

## Primary upstream foundations

- X-Gaussian, `https://github.com/caiyuanhao1998/X-Gaussian`, used as the original reconstruction framework. Confirm its current repository license/permission terms before redistribution; a root license was not present in the reviewed snapshot.
- 3D Gaussian Splatting, `https://github.com/graphdeco-inria/gaussian-splatting`, cited by the upstream project.
- SAX-NeRF, `https://github.com/caiyuanhao1998/SAX-NeRF`, cited by the upstream project.
- Agnes Harness, `https://github.com/AgnesAI-Labs/agnes-harness`, used as the agent runtime and distributed under its own Apache-2.0 license.

## Bundled/native submodules

`submodules/diff-gaussian-rasterization` includes its own `LICENSE.md`. Other bundled or vendored components must retain their original notices and be reviewed individually before public redistribution.

## Competition disclosure

The Vasc-Gaussian model, training code, vessel-aware losses, priors, and evaluation utilities predate the competition integration. New integration work includes the AGH plugin, safe profile bridge, approval and execution workflow, provenance/evidence capture, verification/comparison tools, exception policy, Skill, and competition documentation.

Dataset redistribution rights are separate from source-code rights. Raw CT/X-ray/pickle data, trained weights, private traces, and API keys must not be published unless their licenses and consent allow it.
