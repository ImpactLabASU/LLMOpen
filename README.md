# LLMOpen — Personalized Open-World Safety Planning with LLMs

> Reference implementation and model artifacts for **“Personalized open-world plan generation for safety-critical human-centered autonomous systems: A case study on Artificial Pancreas” (EMNLP Findings 2025)**.

---

## 📌 What’s in this repository?

This repository provides a Hugging Face–compatible language model fine-tuned for **personalized plan suggestion** in safety-critical domains (e.g., insulin management for Artificial Pancreas). It also includes training metadata and adapter files exported from AutoTrain.

Contents:
- **Model files**: `config.json`, `pytorch_model.bin`, `tokenizer.*`
- **Adapter files**: `adapter_config.json`, `adapter_model.bin`
- **Training metadata**: `training_args.bin`, `training_params.json`

These files indicate a **Transformers-based causal language model** with optional **LoRA/PEFT adapters**.

---

## 🧠 Paper at a glance

The accompanying paper introduces an **LLM-based open-world planning framework** for **Human-Centered Autonomous Systems (HCAS)** with a case study in **Artificial Pancreas (AID systems)**.

### Core contributions:
- **LLM planning loop**: LLMs generate and refine personalized plans in open-world settings.
- **Contextualization**: Physics- and safety-aware prompts adapt plans to individualized system parameters.
- **Embodied fine-tuning**: Textual instructions are paired with traces from digital twin simulations.
- **Verification loop**: Plans are validated via a safety simulator (forward model of glucose-insulin dynamics), with unsafe plans triggering rapid LLM replanning.
- **Case study**: Personalized AID planning, evaluated via TIR (time-in-range), TBR (time-below-range), and hypoglycemia avoidance.

---

## 🚀 Quickstart: Local inference

Requirements: `transformers>=4.41`, `accelerate`, `torch`, `peft` (optional if using adapter).

### Run full model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

MODEL_DIR = "./"

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
streamer = TextStreamer(tok)

prompt = (
    "You are assisting a user of an automated insulin delivery (AID) system.\n"
    "Context: CGM=85 mg/dL, ISF=50, CIR=0.36 U/g, user plans 30 min interval training in the next hour.\n"
    "Task: Propose a safe, concise usage plan minimizing hypoglycemia risk."
)

inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    gen = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
print(tok.decode(gen[0], skip_special_tokens=True))
```
Load adapter (LoRA/PEFT)
```python

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_DIR = "./"
base_model = "meta-llama/Llama-2-7b-hf"  # replace with the correct base

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
```
🧪 Reproducing evaluations
The evaluation pipeline consists of four stages:

1. Physics model recovery (personalization)
Calibrate patient-specific glucose-insulin dynamics using SINDy or EMILY.

Save identified parameters (e.g., theta.json, dyn_params.csv) into data/embodied_prompts/.

```bash

python scripts/recover_dynamics.py --input patient_cgm.csv --output data/embodied_prompts/patient1.json
2. Build embodied prompts
Insert calibrated parameters into planning prompts.
```
Generates JSONL or plain text instruction data for training.

```bash

python scripts/build_embodied_prompts.py \
    --dyn data/embodied_prompts/patient1.json \
    --output data/train_prompts/patient1.jsonl
```
3. Fine-tuning with PEFT/LoRA
Train the model or adapters using Hugging Face PEFT.

```bash

python scripts/finetune.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --train_file data/train_prompts/patient1.jsonl \
    --output_dir model/adapter_patient1 \
    --peft lora \
    --epochs 3 \
    --batch_size 8

```
4. Safety verification
Evaluate generated plans with a safety simulator (e.g., UVA/Padova T1D).

Metrics: TIR (%), TBR (%), time spent <70 mg/dL, plan iteration count.

```bash

python scripts/verify_plan.py \
    --model model/adapter_patient1 \
    --sim sim/t1d/patient1_sim.json \
    --output results/patient1_eval.csv

```
Example output:

Patient	TIR (%)	TBR (%)	Hypo events	Iterations
P1	76.2	1.5	0	2

Suggested project layout
```arduino

LLMOpen/
├─ model/                     # model + adapter weights
├─ data/
│  ├─ embodied_prompts/
│  ├─ train_prompts/
│  └─ examples/
├─ sim/
│  └─ t1d/                    # simulator configs + wrappers
├─ scripts/
│  ├─ recover_dynamics.py
│  ├─ build_embodied_prompts.py
│  ├─ finetune.py
│  └─ verify_plan.py
├─ results/
│  └─ patient1_eval.csv
├─ notebooks/                 # demo notebooks
├─ LICENSE
└─ README.md
```
⚙️ Environment
```bash

conda create -n llmopen python=3.10 -y
conda activate llmopen
pip install "transformers>=4.41" accelerate peft torch
# optional tools:
pip install numpy scipy pandas matplotlib sentencepiece datasets evaluate
GPU users: install CUDA-compatible torch wheels from PyTorch.
```
🔍 Example prompts
Exercise (novel action):

```nginx

I plan 30 mins of interval training in 60 mins. Current CGM 85 mg/dL; ISF 50; CIR 0.36.
Propose a safe plan (setpoint, snack yes/no & grams, insulin actions), minimize hypoglycemia.
Plan invalidation (dessert):
```
```css

It’s 6 pm; CGM 121 mg/dL; I had a snack at 3 pm. I want a quarter of a 0.5 lb tiramisu.
Suggest portion and insulin (if any) to keep CGM ≤ 180 mg/dL.
Pregnancy adaptation:
```
```vbnet

Week 6 pregnancy. Suggest day-long meal + exercise plan to keep TIR > 70%,
assuming increased insulin resistance and morning variability.
```

📚 Citation
If you use this code/model, please cite:

```arduino

Banerjee, A., & Gupta, S.K.S. (2025).
Personalized open-world plan generation for safety-critical human-centered autonomous systems:
A case study on Artificial Pancreas. EMNLP Findings 2025.
```
🔒 License
Please add a license file (MIT, BSD-3-Clause, or Apache-2.0 recommended).
Until a license is included, usage rights are undefined.

🙏 Acknowledgments
This work builds on:

Physics-guided model recovery (SINDy, EMILY)

Hugging Face Transformers + AutoTrain

PEFT/LoRA adaptation

UVA/Padova T1D safety simulator

We acknowledge NSF, NIH, DARPA FIRE, and Helmsley Charitable Trust projects supporting this research.


---

