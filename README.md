# Analog Entropy Override Controller (LTSpice)

This module simulates a real-time hazard management circuit using entropy, noise, and ML-based trigger signals to issue FLUSH or LOCK signals during pipeline operation.

## 🧠 Motivation

Inspired by hybrid AI–hardware pipeline control models, this analog overlay introduces a low-latency entropy override path, simulating FSM behavior with physical comparators and logic gates.

## ⚙️ Components

- **Entropy Signal (V_entropy):** Modeled via PWL ramp, represents system disorder.
- **ML Trigger (V_ml_trigger):** Emulates neural prediction override (e.g., `ML=11`).
- **Noise Source (V_noise):** Injects signal ambiguity into the control system.
- **Comparators (OP07):** For entropy and noise evaluation.
- **NAND + Inverters:** Combinational override logic for `LOCK_OUT` and `FLUSH_OUT`.

## 🔁 Circuit Behavior

| Input | Description |
|-------|-------------|
| `V_entropy` | Linearly rising entropy (0 → 5V) |
| `V_ml_trigger` | ML prediction override (0 → 1V at 0.5ms) |
| `V_noise` | Ambient signal perturbation (random or flat) |

| Output | Description |
|--------|-------------|
| `LOCK_OUT` | Activates when both entropy and ML trigger pass threshold |
| `FLUSH_OUT` | Activates under moderate entropy or early hazard detection |

## 📈 Simulation Output

![Simulation Graph](./waveform_output.png)

- `LOCK_OUT` triggers post-entropy ramp when ML trigger is active.
- `FLUSH_OUT` activates slightly earlier, confirming cascade override logic.

## 📂 Files

| File | Description |
|------|-------------|
| `3_input_analog_entropy_override.asc` | Main LTSpice circuit schematic |
| `README.md` | Project summary |
| `LTSpice_Analog_Override_Report.docx` | Full write-up with expected behavior |
| `waveform_output.png` | Simulation waveform image |

## 🧪 Usage

Open the `.asc` file in LTSpice and run a transient analysis (1ms) to observe `FLUSH_OUT` and `LOCK_OUT` response behavior based on entropy and ML trigger thresholds.

## 🚀 Relevance

This simulation prototype complements our Verilog-based FSM pipeline controller and provides analog fallback logic for catastrophic state recovery — ideal for hardware-augmented AI systems.


