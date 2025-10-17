import pandas as pd
from pathlib import Path

# Paths
raw_file   = "data/raw_data/occurrences.csv"
output_jl  = Path("data/maib-incident-reports-dataset.jsonl") # maib-incident-reports-dataset
output_jl.parent.mkdir(parents=True, exist_ok=True)

# --- Load ---
df = pd.read_csv(raw_file, sep=";", quotechar='"', encoding="utf-8-sig", engine="python")

# --- Keep only required columns ---
cols = ["Occurrence_Id", "Short_Description", "Description", "Main_Event_L1"]
df = df[cols].copy()

# --- Build text (no semantic changes) ---
df["text"] = df["Short_Description"].fillna("").astype(str) + " " + df["Description"].fillna("").astype(str)
df = df.rename(columns={"Main_Event_L1": "label"})
df = df.rename(columns={"Occurrence_Id": "id"})

# 1) Unescape slashes and normalize whitespace/newlines
def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"\\/", "/", regex=True)        # "\/" -> "/"
         .str.replace(r"[\r\n]+", " ", regex=True)    # newlines -> space
         .str.replace(r"\s+", " ", regex=True)        # collapse spaces
         .str.replace("“", '"').str.replace("”", '"') # smart quotes
         .str.replace("’", "'").str.replace("–", "-") # apostrophe/dash
         .str.strip()
    )

def shorten_id(series: pd.Series) -> pd.Series:
    """
    Replace long UUID-style IDs with sequential integers starting from 0.
    Returns a new Series with the same index.
    """
    # Create a mapping from old ID → new integer
    id_map = {old_id: i for i, old_id in enumerate(series.unique())}
    return series.map(id_map)

df["id"] = shorten_id(df["id"])
df["text"]  = clean_text(df["text"])
df["label"] = (
    df["label"].astype("string")
              .str.replace(r"\\/", "/", regex=True)      # "\/" -> "/"
              .str.replace(r"\s*/\s*", " / ", regex=True) # tidy spaces around "/"
              .str.strip('"').str.strip()                 # stray quotes/space
)

# 2) Drop rows with missing/blank label or trivially short text
df = df.replace({"label": {"": pd.NA}})
df = df.dropna(subset=["label"])
df = df[df["text"].str.len() > 20]

# 3) Deduplicate (keep one per incident; also drop exact duplicate texts)
df = df.drop_duplicates(subset=["id"], keep="first")
df = df.drop_duplicates(subset=["text"], keep="first")

# --- Keep final fields & save JSONL ---
df = df[["text", "label"]]
df.to_json(output_jl, orient="records", lines=True, force_ascii=False)

print(f"Clean JSONL saved to: {output_jl} (rows={len(df)})")
print("Label samples:", df['label'].drop_duplicates().head(10).tolist())
print("\nSample rows:")
print(df.sample(min(3, len(df)), random_state=42))
