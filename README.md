# ARCHON Hybrid Entropy-Aware Control System (v2.3)

This module (`fsm_entropy_overlay.v`) implements a **multi-priority, analog-digital control unit** for CPU pipeline hazard management. It integrates:

- 🧠 **Machine Learning Predictions**: Guides control flow with predictions (OK, STALL, FLUSH, LOCK).
- 🔀 **Entropy Score Thresholding**: Triggers control actions based on internal digital entropy (`internal_entropy_score`).
- 🔌 **Analog Override System (LTSpice)**: Feeds in **real-time analog signals** for entropy-based override classification.

---

## 🔁 FSM Overview

### States:
- `00`: STATE_OK — Normal operation.
- `01`: STATE_STALL — Pipeline temporarily halted.
- `10`: STATE_FLUSH — Pipeline reset to avoid corrupted instruction propagation.
- `11`: STATE_LOCK — System-critical failure state; must be externally reset.

### Inputs:
- `ml_predicted_action [1:0]` — ML-based hazard classification.
- `internal_entropy_score [7:0]` — Calculated entropy score from QED.
- `internal_hazard_flag` — Hazard detection flag from digital circuit logic.
- `analog_lock_override` — Direct LOCK signal from analog controller.
- `analog_flush_override` — Direct FLUSH signal from analog controller.
- `analog_entropy_severity [1:0]` — Encoded entropy tier from LTSpice analog system:

| Value | Meaning               |
|-------|------------------------|
| 00    | Normal                |
| 01    | Elevated Entropy → STALL |
| 10    | Critical Entropy → FLUSH |

### Outputs:
- `fsm_state [1:0]` — Current FSM control output to pipeline.
- `entropy_log_out [7:0]` — Snapshot of entropy at each transition (for debug/monitoring).

---

## 🔬 System Logic Flow

```verilog
Priority: analog_lock_override > analog_flush_override > analog_entropy_severity > ML prediction > digital hazard
```

### Example Behavior:

- If `analog_lock_override = 1`, system immediately enters `STATE_LOCK`.
- If `analog_flush_override = 1`, system overrides ML and flushes.
- If `analog_entropy_severity = 2'b10`, flushes pipeline preemptively.
- If ML says OK but entropy is high → STALL.
- If no threats detected → remain in `STATE_OK`.

---

## 🛠️ Deployment

### Simulation Tools:
- `Quartus Prime` (Intel FPGA Toolchain)
- `ModelSim` (Verilog simulation)
- `LTSpice` (Analog entropy simulation with comparators, voltage triggers, entropy noise)

### Integration:
- FSM interfaces with analog signals via logic-level digital pins (`LOCK_OUT`, `FLUSH_OUT`, `N_entropy_out` → encoded).
- Can be deployed on Cyclone IV FPGA or similar.

---

## 🔍 Future Extensions

- Add **auto-scaling thresholds** based on pipeline performance metrics.
- Extend analog override to support **LOCK escalation on entropy + noise combo**.
- Publish this as **Paper 5**: "Hybrid Analog–Digital Control for Entropy-Aware Pipeline Architectures"

---
📬 Want to collaborate?  
I'm looking for lab/startup partners for Fall 2025.  
Reach out via [LinkedIn](https://www.linkedin.com/in/joshua-carter-898868356/) or [joshuathomascarter@gmail.com](mai

