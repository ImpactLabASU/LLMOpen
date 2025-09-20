# LLMOpen — Personalized Open-World Safety Planning with LLMs

> Reference implementation and model artifacts for **“Personalized open-world plan generation for safety-critical human-centered autonomous systems: A case study on Artificial Pancreas” (EMNLP Findings 2025)**.

---

## 📌 What’s in this repository?

This repo hosts a Hugging Face–compatible language model fine-tuned for **plan suggestion** in safety-critical settings (e.g., insulin management) alongside training metadata exported from AutoTrain.

Contents:
- `config.json`, `pytorch_model.bin`, `tokenizer.*`
- `adapter_config.json`, `adapter_model.bin`
- `training_args.bin`, `training_params.json`

These indicate an **HF Transformers text-generation model** with an optional **adapter (LoRA/PEFT)**.

---

## 🧠 Paper at a glance

The work introduces an **LLM-based planning pipeline** for open-world, safety-critical HCAS (Human-Centered Autonomous Systems). Core ideas:

- Use an LLM to propose usage plans.
- **Contextualize** with physics-aware prompts and **fine-tune** on embodied data (e.g., SINDy-/EMILY-derived parameters).
- **Verify** each plan via a safety simulator; feed the score back to the LLM for rapid re-planning.
- Case study: **Artificial Pancreas** planning with metrics like TIR/TBR and hypoglycemia avoidance.

---

## 🚀 Quickstart: Local inference

Requirements: `transformers>=4.41`, `accelerate`, `torch`, `peft` (optional).

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
    "Task: Propose a safe, concise usage plan, minimizing hypoglycemia risk."
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
