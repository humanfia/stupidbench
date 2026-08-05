# stupidbench

> A weekly bench that gives coding agents twenty-four hours each to make one
> kernel faster, and reports how far each one got per dollar, per token, and per
> hour.

Every flow — a coding-agent CLI on one model — is looped over the same task on
three seeds. The task is to lower the cycle count of a kernel, scored by an
evaluator the agent can reach but cannot change. It runs entirely on GitHub
Actions: nothing is set up by hand, and every run publishes what its agents did.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Results](#results)
- [Security](#security)
- [License](#license)

## Install

```sh
uv sync
```

Cells run in Docker, so a host needs Docker Engine 28 or newer. The image a
cell runs in is `ghcr.io/humanfia/flowbench-runtime`, by the variant its task
calls for; it is public and nothing here builds it. Each cell is given the token
of the CLI it runs — `CLAUDE_CODE_OAUTH_TOKEN`, `CODEX_ACCESS_TOKEN` or
`KIMI_MODEL_API_KEY` — from the environment.

## Usage

The bench runs itself: [`stupid bench`](.github/workflows/bench.yaml) starts
every six hours on Sundays and Mondays, gives each cell as much of its
twenty-four hours as a job can hold, and hands what it has to the next one. Run
it early from the Actions tab.

The same four commands work on any host:

```sh
stupidbench prepare --flow opus5_max --seed 0   # stage one cell
stupidbench run --flow opus5_max --seed 0 --seconds 3600
stupidbench redact                              # take every credential back out
stupidbench report --out report                 # draw the curves, write the report
```

`prepare` leaves the work of a cell that is already staged alone and `run` picks
a cell up where it was left, so repeating either continues rather than restarts. A cell is done
when it has spent its twenty-four hours of agent time; time between runs is not
charged to it.

Flows are `gpt56sol_max`, `gpt56terra_max`, `gpt56luna_max`, `opus5_max` and
`k3_max`; seeds are 0, 1 and 2.

## Results

Every run leaves:

- **a report** in the job summary — best score per flow, and per cell, against
  what each spent;
- **the curves**, as a `report` artifact: score against dollars, output tokens
  and agent hours, with the three seeds of a flow averaged;
- **every trajectory**, as one `cell-<flow>-<seed>` artifact per cell — the
  sessions its CLI kept, its scores, its events, and the work it leaves behind.

## Security

This repository is public, so its artifacts, caches and logs are too. Nothing
that runs here is reachable from a pull request: only the schedule and a manual
dispatch can carry a token, and both need write access.

A cell is given exactly one credential — the token of the CLI it runs, never the
others — and reaches the network only through a proxy that allows the model
providers and denies everything else. It can look up nothing else either, so not
even a hostname it asks for leaves carrying something, and a token cannot be
sent anywhere it is not already used. Before anything leaves a runner,
`stupidbench redact` deletes
what the CLIs store credentials in and replaces every occurrence of a live
token, or of anything shaped like one, anywhere an agent may have written it. A
segment that cannot redact publishes nothing.

## License

Unlicensed.
