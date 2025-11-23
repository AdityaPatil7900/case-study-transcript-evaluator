# score_engine.py
"""
Score engine for the case study rubric + transcripts.

Features:
- Robust rubric loader from Excel or CSV (handles many column name variants).
- Rule-based scoring (keyword presence + length checks).
- Optional semantic scoring using sentence-transformers (graceful fallback if not installed).
- Batch scoring helpers and CSV/JSON export helpers.
- Example runner using default local files (adjust paths as needed).

Author: Aditya Patil
"""

import os
import re
import json
import math
import warnings
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
import numpy as np

# Try to import sentence-transformers & sklearn cosine similarity.
# If not available, semantic scoring will be disabled but rest of pipeline works.
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    try:
        from sklearn.metrics.pairwise import cosine_similarity  # attempt to keep function if sklearn present
    except Exception:
        def cosine_similarity(a, b):
            raise RuntimeError("cosine_similarity not available (sklearn is missing)")

# ---------------------------
# Configuration / defaults
# ---------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA = 0.5   # weight for rule-based score
BETA = 0.5    # weight for semantic score
KEYWORD_WEIGHT = 0.9   # within rule_score: fraction for keywords vs length
LENGTH_WEIGHT = 0.1

# Default local paths (update to your project folder if needed)
DEFAULT_RUBRIC_PATH = r"D:\PROJECTS\evaluates a student’s spoken introduction using the given transcript\rubric_clean.csv"
DEFAULT_SAMPLE_PATH = r"D:\PROJECTS\evaluates a student’s spoken introduction using the given transcript\Sample text for case study.txt"

# ---------------------------
# Utilities
# ---------------------------
def clean_text(t: Any) -> str:
    """Normalize text safely (handles NaN)."""
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    s = str(t)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_keywords(k: Any) -> List[str]:
    """Parse a keywords cell into a list of lowercase keyword strings."""
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return []
    if isinstance(k, (list, tuple)):
        return [str(x).strip().lower() for x in k if str(x).strip()]
    s = str(k)
    parts = re.split(r'[,\n;]+', s)
    return [p.strip().lower() for p in parts if p.strip()]

# ---------------------------
# Rubric loader (CSV or Excel) - flexible
# ---------------------------
def load_rubric(path_or_df: Any) -> pd.DataFrame:
    """
    Load rubric from a CSV/Excel path or accept a pre-loaded DataFrame.
    Returns a normalized DataFrame with columns:
      ['criterion', 'description', 'keywords', 'weight', 'min_words', 'max_words']
    The loader is flexible to various header names.
    """
    # If the user passed a DataFrame already, normalize it
    if isinstance(path_or_df, pd.DataFrame):
        df_raw = path_or_df.copy()
    else:
        if not isinstance(path_or_df, str):
            raise ValueError("load_rubric expects a file path (str) or a pandas.DataFrame.")
        path = path_or_df
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rubric file not found at: {path}")
        _, ext = os.path.splitext(path.lower())
        if ext == ".csv":
            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            # Try Excel
            df_raw = pd.read_excel(path, engine='openpyxl', dtype=str)

    # If dataframe is empty
    if df_raw.shape[0] == 0 or df_raw.shape[1] == 0:
        raise ValueError("Rubric file appears empty or incorrectly formatted.")

    # Normalize column names for heuristic detection
    orig_columns = list(df_raw.columns)
    norm_columns = {c: str(c).strip().lower().replace(" ", "_") for c in orig_columns}

    # Attempt to find candidate header columns
    col_map: Dict[str, str] = {}

    for orig, norm in norm_columns.items():
        if any(k in norm for k in ["criterion", "criteria", "creteria", "parameter", "evaluation", "eval"]):
            col_map.setdefault("criterion", orig)
        if any(k in norm for k in ["description", "detail", "explanation", "note"]):
            col_map.setdefault("description", orig)
        if "keyword" in norm or "key_word" in norm:
            col_map.setdefault("keywords", orig)
        if any(k in norm for k in ["weight", "weightage", "score"]):
            col_map.setdefault("weight", orig)
        if "min" in norm and "word" in norm:
            col_map.setdefault("min_words", orig)
        if "max" in norm and "word" in norm:
            col_map.setdefault("max_words", orig)

    # If we didn't detect a criterion column, fallback to the first non-empty string-like column
    if "criterion" not in col_map:
        for c in orig_columns:
            if df_raw[c].astype(str).str.strip().replace("", pd.NA).notna().any():
                col_map["criterion"] = c
                break

    # Provide defaults for columns not found
    if "keywords" not in col_map:
        df_raw["keywords"] = ""
        col_map["keywords"] = "keywords"
    if "description" not in col_map:
        df_raw["description"] = ""
        col_map["description"] = "description"
    if "weight" not in col_map:
        df_raw["weight"] = 1.0
        col_map["weight"] = "weight"

    # Build normalized list of rubric rows
    rows = []
    for _, row in df_raw.iterrows():
        crit = clean_text(row.get(col_map.get("criterion")))
        if crit == "":
            # skip blank criterion rows
            continue
        desc = clean_text(row.get(col_map.get("description")))
        kws = parse_keywords(row.get(col_map.get("keywords")))
        # parse weight safely
        w_raw = row.get(col_map.get("weight"))
        try:
            weight = float(str(w_raw).strip()) if w_raw is not None and str(w_raw).strip() != "" else 1.0
        except Exception:
            weight = 1.0
        # try parsing min/max words if columns present
        min_w = None
        max_w = None
        if col_map.get("min_words") in row and row.get(col_map.get("min_words")) not in (None, "", np.nan):
            try:
                min_w = int(float(str(row.get(col_map.get("min_words"))).strip()))
            except Exception:
                min_w = None
        if col_map.get("max_words") in row and row.get(col_map.get("max_words")) not in (None, "", np.nan):
            try:
                max_w = int(float(str(row.get(col_map.get("max_words"))).strip()))
            except Exception:
                max_w = None

        rows.append({
            "criterion": crit,
            "description": desc,
            "keywords": kws,
            "weight": weight,
            "min_words": min_w,
            "max_words": max_w
        })

    rubric_df = pd.DataFrame(rows)
    if rubric_df.shape[0] == 0:
        raise ValueError("No rubric rows parsed. Please inspect the input file or pass the cleaned CSV produced by parse_rubric_custom.py")
    # Optional: reset index
    rubric_df = rubric_df.reset_index(drop=True)
    return rubric_df

# ---------------------------
# Embedding model (lazy loaded)
# ---------------------------
class EmbedModel:
    _model = None

    @classmethod
    def get_model(cls):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed. Install it to enable semantic scoring.")
        if cls._model is None:
            cls._model = SentenceTransformer(EMBED_MODEL_NAME)
        return cls._model

# ---------------------------
# Scoring helpers
# ---------------------------
def rule_score_for_criterion(text: str, keywords: List[str], min_words: Optional[int]=None, max_words: Optional[int]=None) -> Tuple[float, Dict[str,Any]]:
    """Rule-based score using keyword coverage and length penalty."""
    t = clean_text(text).lower()
    words = t.split()
    nwords = len(words)
    found = []
    for kw in keywords:
        kw_clean = str(kw).strip().lower()
        if not kw_clean:
            continue
        # containment check (simple); can be improved with regex/lemmatization later
        if kw_clean in t:
            found.append(kw_clean)
    total_k = max(1, len(keywords))
    kw_frac = len(found) / total_k
    length_score = 1.0
    length_feedback = None
    if min_words and nwords < min_words:
        length_score = nwords / (min_words + 1e-9)
        length_feedback = f"Too short ({nwords} words, min {min_words})"
    elif max_words and nwords > max_words:
        length_score = max(0.0, 1.0 - (nwords - max_words) / (max_words + 1e-9))
        length_feedback = f"Too long ({nwords} words, max {max_words})"
    rule_score = KEYWORD_WEIGHT * kw_frac + LENGTH_WEIGHT * length_score
    diagnostics = {
        "keywords_found": found,
        "keywords_total": len(keywords),
        "kw_frac": kw_frac,
        "nwords": nwords,
        "length_score": length_score,
        "length_feedback": length_feedback
    }
    return float(rule_score), diagnostics

def semantic_score_for_criterion(text: str, criterion_desc: str) -> Tuple[float, float]:
    """Semantic similarity score [0,1] using embeddings. Returns (score, raw_cosine)."""
    if not _HAS_ST:
        return 0.0, 0.0
    try:
        model = EmbedModel.get_model()
        desc_emb = model.encode([clean_text(criterion_desc)], convert_to_numpy=True)
        text_emb = model.encode([clean_text(text)], convert_to_numpy=True)
        cos = float(cosine_similarity(desc_emb, text_emb)[0,0])
        score = (cos + 1.0) / 2.0
        return float(score), float(cos)
    except Exception as e:
        warnings.warn(f"Semantic scoring error: {e}")
        return 0.0, 0.0

# ---------------------
# Core scoring function
# ---------------------------
def score_transcript(transcript_text: str, rubric_df: pd.DataFrame, alpha: float=ALPHA, beta: float=BETA) -> Dict[str, Any]:
    """
    Score one transcript. Returns dict:
      {
        "overall_score": float (0-100),
        "per_criterion": [ {criterion, description, keywords_found, ...}, ... ],
        "alpha": alpha, "beta": beta, "model_enabled": bool
      }
    """
    t = clean_text(transcript_text)
    per = []
    if 'weight' in rubric_df.columns:
        total_weight = float(rubric_df['weight'].replace(0, np.nan).dropna().sum()) if len(rubric_df)>0 else 1.0
        if not total_weight or math.isnan(total_weight):
            total_weight = float(len(rubric_df))
    else:
        total_weight = float(len(rubric_df))

    overall_raw = 0.0

    for _, row in rubric_df.iterrows():
        crit = row.get('criterion', '') or ''
        desc = row.get('description', '') or crit
        keywords = row.get('keywords') or []
        weight = float(row.get('weight') or 1.0)
        min_w = row.get('min_words', None)
        max_w = row.get('max_words', None)

        rule_sc, rule_diag = rule_score_for_criterion(t, keywords, min_w, max_w)
        sem_sc, raw_cos = semantic_score_for_criterion(t, desc)

        raw = float(alpha * rule_sc + beta * sem_sc)
        weighted_raw = raw * (weight / (total_weight + 1e-12))
        overall_raw += weighted_raw

        per.append({
            "criterion": crit,
            "description": desc,
            "keywords_found": rule_diag["keywords_found"],
            "keywords_total": rule_diag["keywords_total"],
            "nwords": rule_diag["nwords"],
            "length_feedback": rule_diag["length_feedback"],
            "rule_score": round(rule_sc, 4),
            "semantic_score": round(sem_sc, 4),
            "raw_cosine": round(raw_cos, 4),
            "combined_raw": round(raw, 4),
            "weight": weight,
            "weighted_raw": round(weighted_raw, 6)
        })

    overall_score = float(max(0.0, min(1.0, overall_raw))) * 100.0
    return {
        "overall_score": round(overall_score, 2),
        "per_criterion": per,
        "alpha": alpha,
        "beta": beta,
        "model_enabled": _HAS_ST
    }

# ---------------------------
# Batch helpers
# ---------------------------
def batch_score_transcripts(transcripts: List[Dict[str,Any]], rubric_df: pd.DataFrame, out_csv: Optional[str]=None, out_json: Optional[str]=None) -> List[Dict[str,Any]]:
    """
    transcripts: [{'id':..., 'text':...}, ...]
    Returns list of results and optionally writes CSV/JSON.
    """
    results = []
    for t in transcripts:
        res = score_transcript(t.get('text', ''), rubric_df)
        results.append({"id": t.get('id', None), "overall_score": res["overall_score"], "per_criterion": res["per_criterion"]})
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    if out_csv:
        rows = []
        for r in results:
            for pc in r["per_criterion"]:
                rows.append({
                    "id": r["id"],
                    "overall_score": r["overall_score"],
                    "criterion": pc["criterion"],
                    "rule_score": pc["rule_score"],
                    "semantic_score": pc["semantic_score"],
                    "combined_raw": pc["combined_raw"],
                    "weight": pc["weight"],
                    "keywords_found": ";".join(pc["keywords_found"]) if pc["keywords_found"] else "",
                    "keywords_total": pc["keywords_total"],
                    "nwords": pc["nwords"],
                    "length_feedback": pc["length_feedback"]
                })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return results

# ------------------------
# CLI / example runner
# ---------------------------
if __name__ == "__main__":
    rubric_path = DEFAULT_RUBRIC_PATH
    sample_path = DEFAULT_SAMPLE_PATH

    print("Using rubric:", rubric_path)
    if not os.path.exists(rubric_path):
        print("Rubric file not found at default path. Please update DEFAULT_RUBRIC_PATH in the script.")
    else:
        try:
            rubric = load_rubric(rubric_path)
            print("Loaded rubric with", len(rubric), "criteria.")
            print(rubric[['criterion','keywords','weight','min_words','max_words']].head().to_string(index=False))
        except Exception as e:
            print("Failed to load rubric:", e)
            raise

        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                txt = f.read()
            print("\nScoring sample transcript...")
            res = score_transcript(txt, rubric)
            print("Overall score:", res['overall_score'])
            print("Per-criterion summary:")
            for pc in res['per_criterion']:
                print(f"- {pc['criterion']}: combined={pc['combined_raw']}, keywords_found={pc['keywords_found']}, words={pc['nwords']}")
            with open("sample_result.json", "w", encoding="utf-8") as of:
                json.dump(res, of, indent=2, ensure_ascii=False)
            print("\nSaved sample_result.json")
        else:
            print("Sample transcript not found at default sample path:", sample_path)
