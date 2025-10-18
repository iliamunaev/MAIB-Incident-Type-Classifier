import pandas as pd

occurrences = "data/raw_data/occurrences.csv"

print(f"Reading file: {occurrences}\n")

df = pd.read_csv(
    occurrences,
    sep=";",
    quotechar='"',
    encoding="utf-8-sig",
    engine="python",
    na_values=["", "NA"],   
)

print(df.head())
print("\nColumns:", df.columns.tolist())
# print("Shape:", df.shape, "\n")
