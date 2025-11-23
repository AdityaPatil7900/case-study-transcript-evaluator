# run this in your project (VS Code / Anaconda Prompt)
import pandas as pd
import re, os

path = r"D:\PROJECTS\evaluates a student’s spoken introduction using the given transcript\Case study for interns.xlsx"
# path = r"/mnt/data/Case study for interns.xlsx"   # alternate uploaded path if you prefer

xls = pd.ExcelFile(path, engine='openpyxl')
sheet = xls.sheet_names[0]
print("Sheet:", sheet)

# read without header to inspect
df_raw = pd.read_excel(path, sheet_name=sheet, header=None, engine='openpyxl')
pd.set_option('display.max_colwidth', 200)
print("\n--- Top 20 rows (for inspection) ---")
print(df_raw.head(20).to_string(index=True, header=False))

# keywords that likely appear in header row
header_keywords = ["metric", "weight", "weightage", "criteria", "criterion", "creteria", "keyword", "keywords", "description", "salutation", "overall rubrics"]

candidates = []
for r in range(min(30, df_raw.shape[0])):
    row_text = " ".join([str(x).lower() for x in df_raw.iloc[r].fillna("").astype(str).values])
    if any(k in row_text for k in header_keywords):
        candidates.append(r)

print("\nHeader row candidates found (0-based indices):", candidates)

if not candidates:
    # fallback heuristic: find first row that has at least two non-empty cells
    for r in range(min(30, df_raw.shape[0])):
        if df_raw.iloc[r].count() >= 2:
            candidates = [r]
            break

header_row = candidates[0]
print("Using header_row =", header_row)

# re-read using that header row
df = pd.read_excel(path, sheet_name=sheet, header=header_row, engine='openpyxl')

# clean column names
df.columns = [str(c).strip() for c in df.columns]

print("\nDetected columns after reloading with header:")
print(df.columns.tolist())

# Show first few rows of content (these should be rubric rows)
print("\nFirst 10 data rows after header (for manual check):")
print(df.head(10).to_string(index=False))

# Try to auto-detect rubric columns (best-effort)
cols = [c.lower() for c in df.columns]
mapping = {}
for i,c in enumerate(df.columns):
    cl = c.lower()
    if any(k in cl for k in ["criterion","criteria","creteria","parameter","creteria"]):
        mapping['criterion'] = c
    if any(k in cl for k in ["keyword","keywords","key word"]):
        mapping['keywords'] = c
    if any(k in cl for k in ["weight","weightage","score"]):
        mapping['weight'] = c
    if any(k in cl for k in ["description","detail","explanation"]):
        mapping['description'] = c
    if "min" in cl and "word" in cl:
        mapping['min_words'] = c
    if "max" in cl and "word" in cl:
        mapping['max_words'] = c

print("\nAuto-detected column mapping (best-effort):")
print(mapping)

# Build normalized rubric DF (use detected mapping; if missing, create defaults)
rows = []
for _, r in df.iterrows():
    crit = r.get(mapping.get('criterion')) if mapping.get('criterion') else None
    if pd.isna(crit) or str(crit).strip()=="":
        # skip blank rows
        continue
    desc = r.get(mapping.get('description')) if mapping.get('description') else ""
    kws = r.get(mapping.get('keywords')) if mapping.get('keywords') else ""
    weight = r.get(mapping.get('weight')) if mapping.get('weight') else 1.0
    # parse keywords as comma separated if string
    if isinstance(kws, str):
        kw_list = [x.strip().lower() for x in re.split(r'[,\n;]+', kws) if x.strip()]
    else:
        kw_list = []
    try:
        weight = float(weight)
    except:
        weight = 1.0
    rows.append({
        "criterion": str(crit).strip(),
        "description": str(desc).strip() if not pd.isna(desc) else "",
        "keywords": kw_list,
        "weight": weight,
        "min_words": None,
        "max_words": None
    })

rubric_df = pd.DataFrame(rows)
print("\nFinal parsed rubric preview:")
print(rubric_df.head(20).to_string(index=False))

# Save cleaned rubric if you want
out_csv = os.path.join(os.path.dirname(path), "rubric_clean.csv")
rubric_df.to_csv(out_csv, index=False)
print("\nSaved cleaned rubric to:", out_csv)

