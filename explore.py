import pandas as pd

occurrences = "data/raw_data/occurrences.csv"
# vessels = "/kaggle/working/vessels.csv"
# affected_persons = "/kaggle/working/affected_persons.csv"

print("***************************************\n")
print(f"Reading file: {occurrences}\n")

df = pd.read_csv(
    occurrences,
    sep=";",                # semicolon delimiter
    quotechar='"',          # quoted text fields
    encoding="utf-8-sig",   # handle BOM
    engine="python",
    na_values=["", "NA"],   # interpret blanks/NA
)

print(df.head())
print("\nColumns:", df.columns.tolist())
# print("Shape:", df.shape, "\n")
