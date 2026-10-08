import json
from datetime import datetime, timezone

LOG_FILE = "day_15_adversarial/guardrail_decisions.jsonl"


def log_guardrail(case_id, control, outcome, reason_code):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "control": control,
        "outcome": outcome,
        "reason_code": reason_code
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")