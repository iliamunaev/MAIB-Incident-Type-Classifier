import os, json, numpy as np
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
import evaluate

set_seed(42)

# ---------- Kaggle env checks ----------
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability())
else:
    print("Running on CPU")

# ---------- Load dataset ----------
ds = load_dataset("json", data_files="/kaggle/input/maib-raw/maib_raw.jsonl")["train"]

# ---------- Train/val/test split ----------
ds = ds.train_test_split(test_size=0.2, seed=42)
tmp = ds["test"].train_test_split(test_size=0.5, seed=42)
ds = {"train": ds["train"], "validation": tmp["train"], "test": tmp["test"]}

# ---------- Encode labels with ClassLabel ----------
labels_sorted = sorted(list(set(ds["train"]["label"])))
label_feature = ClassLabel(names=labels_sorted)

def encode_label(batch):
    return {"labels": [label_feature.str2int(x) for x in batch["label"]]}

num_proc = max(1, (os.cpu_count() or 2) // 2)  # mild parallelism on Kaggle
for split in ds:
    ds[split] = ds[split].map(encode_label, batched=True, num_proc=num_proc)

# ---------- Tokenizer + preprocessing ----------
model_ckpt = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)

for split in ds:
    ds[split] = ds[split].map(tokenize, batched=True, remove_columns=["text", "label"], num_proc=num_proc)
    ds[split] = ds[split].with_format("torch")  # torch tensors for Trainer

# ---------- Dynamic padding ----------
collator = DataCollatorWithPadding(tok)

# ---------- Model ----------
num_labels = len(labels_sorted)
id2label = {i: l for i, l in enumerate(labels_sorted)}
label2id = {l: i for i, l in enumerate(labels_sorted)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Optional: slight VRAM savings when using fp16
if use_cuda:
    model.gradient_checkpointing_enable()

# ---------- Metrics ----------
acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
    }

# ---------- Training args (tuned for Kaggle T4) ----------
# T4 supports fp16, not bf16 (bf16 is Ampere+)
is_ampere_plus = use_cuda and (torch.cuda.get_device_capability()[0] >= 8)

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,
    per_device_train_batch_size=32 if use_cuda else 16,   # bigger on GPU
    per_device_eval_batch_size=64 if use_cuda else 32,
    num_train_epochs=3,

    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",

    # Dataloader + memory
    dataloader_pin_memory=use_cuda,                       # avoid warning on CPU
    dataloader_num_workers=num_proc,
    eval_accumulation_steps=4 if use_cuda else None,

    # Speedups
    fp16=use_cuda,
    bf16=is_ampere_plus,                                  # False on T4
    gradient_checkpointing=True,
    optim="adamw_torch",
    lr_scheduler_type="linear",
    warmup_ratio=0.1,

    # Optional small boost on PyTorch 2.x
    torch_compile=True if use_cuda else False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,                 # keep tokenizer= for HF 4.x compatibility
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------- Final test evaluation ----------
print(trainer.evaluate(ds["test"]))

# ---------- Save label list ----------
os.makedirs("out", exist_ok=True)
with open("out/labels.json", "w", encoding="utf-8") as f:
    json.dump(labels_sorted, f, ensure_ascii=False, indent=2)
