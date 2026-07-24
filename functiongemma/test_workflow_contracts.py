"""Safety contracts for the FunctionGemma recovery workflows."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_WORKFLOW = ROOT / ".github/workflows/train-functiongemma.yml"
SWEEP_WORKFLOW = ROOT / ".github/workflows/functiongemma-quant-sweep.yml"


def _steps(path: Path, job: str):
    workflow = yaml.safe_load(path.read_text())
    return {step.get("name"): step for step in workflow["jobs"][job]["steps"]}


def test_rejected_attempt_is_remembered_without_becoming_promotion_state():
    workflow = yaml.safe_load(TRAIN_WORKFLOW.read_text())
    configure = workflow["jobs"]["configure"]["steps"][0]
    assert '[[ "$ACK_RUN" =~ ^[0-9]+$ ]]' in configure["run"]
    assert "acknowledge_run_id requires an explicit locale" in configure["run"]
    assert "acknowledge_run_id and force are mutually exclusive" in configure["run"]

    steps = _steps(TRAIN_WORKFLOW, "train")

    acknowledged = steps["Acknowledge prior attempted inputs (no training)"]
    assert "acknowledge_run_id != ''" in acknowledged["if"]
    assert acknowledged["env"]["ACK_RUN"] == "${{ github.event.inputs.acknowledge_run_id }}"
    assert '"workflow_run": "$ACK_RUN"' in acknowledged["run"]
    assert '"gate_outcome": "acknowledged"' in acknowledged["run"]
    assert "last-attempted-${{ matrix.locale }}.json" in acknowledged["run"]

    check = steps["Check for skill changes since last training"]
    assert '[[ -n "$ACK_RUN" ]]' in check["run"]
    assert 'echo "should_train=false"' in check["run"]

    attempted = steps["Store attempted input SHAs"]
    assert "!cancelled()" in attempted["if"]
    assert "steps.manifest.outcome == 'success'" in attempted["if"]
    assert "steps.gate.outputs.exit_code == '1'" in attempted["if"]
    assert "last-attempted-${{ matrix.locale }}.json" in attempted["run"]

    promoted = steps["Store promoted training SHAs"]
    assert "steps.gate.outcome == 'success'" in promoted["if"]
    assert "last-trained-${{ matrix.locale }}.json" in promoted["run"]

    gate = steps["Routing-eval promotion gate (generated set at derived floor)"]
    assert 'echo "exit_code=$STATUS" >> "$GITHUB_OUTPUT"' in gate["run"]


def test_quant_sweep_is_manual_diagnostic_only_and_keeps_gate_bars():
    workflow = yaml.safe_load(SWEEP_WORKFLOW.read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert set(workflow["jobs"]) == {"sweep"}

    steps = _steps(SWEEP_WORKFLOW, "sweep")
    evaluate = steps["Evaluate all variants on the same runner"]
    assert "--precision-min 0.85" in evaluate["run"]
    assert "--abstain-min 0.90" in evaluate["run"]
    for variant in ["r127_q4", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"]:
        assert f'--model "{variant}=' in evaluate["run"]

    text = SWEEP_WORKFLOW.read_text()
    assert "gh release" not in text
    assert "publish_manifest.py" not in text
    assert "modal run functiongemma/modal_train.py" in text
    assert "--quant-sweep" in text
