# no-mistakes gate

This repo is set up for [no-mistakes](https://github.com/kunchenguid/no-mistakes): push through a local validation gate before code reaches GitHub.

## Prerequisites

```powershell
# already installed on this machine (v1.34.0)
no-mistakes doctor
gh auth login
```

`no-mistakes doctor` must show at least one **agent** (`claude`, `codex`, `copilot`, …) and **`gh`** for PR/CI automation.

## One-time init (after `origin` remote exists)

```powershell
cd nexus-forge-cyto
no-mistakes init
```

## Gated push

```powershell
git checkout -b fix/my-change
# commit ...
git push no-mistakes fix/my-change
no-mistakes   # TUI — approve findings, watch test/lint/review
```

Pipeline order: `intent → rebase → review → test → document → lint → push → pr → ci`.

Local test commands exercised by the gate (configure in repo or agent-detected):

- `cargo test --all-targets`
- `python -m pytest tests/ -q`
