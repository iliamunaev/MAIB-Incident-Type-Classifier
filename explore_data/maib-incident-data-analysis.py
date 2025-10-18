#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

print("Step 1: Loading the MAIB incident reports dataset from JSONL format")
print("=" * 60)

df = pd.read_json(
    '../data/maib-incident-reports-dataset.jsonl',
    lines=True
)

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())


# In[ ]:


print("\nStep 2: Examining the last 5 rows of the dataset")
print("=" * 50)
print("Last 5 rows of the dataset:")
print(df.tail())


# In[ ]:


print("\nStep 3: Getting the total number of records in the dataset")
print("=" * 55)
print(f"Total number of incident reports: {len(df)}")


# In[ ]:


print("\nStep 4: Analyzing dataset structure and data types")
print("=" * 50)
print("Dataset information including column names, data types, and memory usage:")
frame = pd.DataFrame(df)
frame.info()


# In[ ]:


print("\nStep 5: Analyzing label distribution in the dataset")
print("=" * 50)
print("Counting occurrences and calculating percentages for each incident type:")

counts = df['label'].value_counts()
percentages = df['label'].value_counts(normalize=True) * 100

summary = pd.DataFrame({
    'Count': counts,
    'Percentage': percentages.round(2)
})

print("\nLabel distribution summary:")
print(summary)


# In[ ]:


print("\nStep 6: Checking for duplicate rows in the entire dataset")
print("=" * 55)
print(f"Number of duplicate rows: {df.duplicated().sum()}")


# In[ ]:


print("\nStep 7: Checking for duplicate text content in the 'text' column")
print("=" * 60)
print(f"Number of duplicate text entries: {df['text'].duplicated().sum()}")


# In[ ]:


print("\nStep 8: Data exploration completed!")
print("=" * 35)
print("Summary of findings:")
print(f"- Total records: {len(df)}")
print(f"- Dataset shape: {df.shape}")
print(f"- Columns: {list(df.columns)}")
print(f"- Duplicate rows: {df.duplicated().sum()}")
print(f"- Duplicate text entries: {df['text'].duplicated().sum()}")
print(f"- Number of unique incident types: {df['label'].nunique()}")

# convert the notebook to a script
# jupyter nbconvert --to script maib-incident-data-analysis.ipynb

