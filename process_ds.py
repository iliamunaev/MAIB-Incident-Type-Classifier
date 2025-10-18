from datasets import load_dataset, ClassLabel, DatasetDict
import os

# Loads the file as a Hugging Face Dataset.
ds = load_dataset("json", data_files="data/maib-incident-reports-dataset.jsonl")["train"]

# Split 80 % for training, 20 % for testing.
# Reproducible, seed=42. BitGenerator provides a stream of random values.
# In order to generate reproducible streams, BitGenerators support setting their initial state via a seed.
# All of the provided BitGenerators will take an arbitrary-sized non-negative integer, or a list of such integers, as a seed.
ds = ds.train_test_split(test_size=0.2, seed=42)

# Split the test set into validation and test.
tmp = ds["test"].train_test_split(test_size=0.5, seed=42)

# Recombine into a tidy dict
ds = {"train": ds["train"], "validation": tmp["train"], "test": tmp["test"]}

# extracts all the label values
# removes duplicates
# sorts labels alphabetically
labels_sorted = sorted(set(ds["train"]["label"]))

# turn "label" column from plain text into an integer-encoded column with class metadata
for split in ["train", "validation", "test"]:
    ds[split] = ds[split].cast_column("label", ClassLabel(names=labels_sorted))

# 3) If your model/Trainer expects column name 'labels', make a numeric copy
def to_labels(batch):
    # after cast, 'label' is already ints; create 'labels' for Trainer
    return {"labels": batch["label"]}

num_proc = max(1, (os.cpu_count() or 2) // 2)
for split in ds:
    ds[split] = ds[split].map(to_labels, batched=True, num_proc=num_proc)

print(labels_sorted)                         # list of class names
print(ds)                                    # DatasetDict with splits
print(ds["train"].features)                  # shows ClassLabel on 'label'
print(ds["train"][0].keys())                 # includes 'text','label','labels'
print(ds["train"].features["label"].names)   # class names
print(ds["train"][0]["label"], ds["train"][0]["labels"])  # both ints, same value


ds = DatasetDict(ds)
ds.save_to_disk("data/maib-incident-reports-5K-processed")
