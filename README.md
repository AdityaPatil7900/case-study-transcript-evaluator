# case-study-transcript-evaluator
📘 Case Study — Transcript Scoring Engine
Automated Rubric-Based Transcript Evaluation (Streamlit + Python)

Author: Aditya Jaypalsing Patil
📧 adityapatil0790@gmail.com

🔗 LinkedIn - https://www.linkedin.com/in/aditya-patil-aj7900/

🚀 Project Overview

This project is a Transcript Evaluation System built for the Nirmaan AI Internship Case Study.

It automatically evaluates a student’s spoken introduction using:

A rubric Excel sheet

A transcript text file

Rule-based scoring (keywords + length)

Optional semantic scoring using embeddings

A Streamlit web app for interactive scoring

🔥 Key Feature:
The project handles extremely messy Excel rubrics, converts them into a clean CSV, and evaluates any transcript using a weighted scoring model.

📁 Repository Structure
case-study-transcript-evaluator/
│
├── app_streamlit.py            → Main Streamlit web app
├── score_engine.py             → Core scoring engine
├── parse_rubric_custom.py      → Clean & parse rubric Excel → rubric_clean.csv
├── run_example.py              → CLI tester (scores sample transcript)
│
├── Case study for interns.xlsx → Original rubric file (input)
├── Sample text for case study.txt → Sample transcript (input)
├── rubric_clean.csv            → Clean rubric produced by parser
├── sample_result.json          → Output example from run_example.py
│
└── README.md                   → This file

🧠 What the Project Does
✅ 1. Reads the Rubric (Even if Messy)

The parser:

Detects headers automatically

Extracts criteria, keywords, weights

Cleans and generates a usable structured rubric

✅ 2. Evaluates Transcripts

Each criterion is scored using:

Keyword presence

Length check (min/max words)

Semantic similarity (optional)

✅ 3. Streamlit Web App

Upload:

Rubric Excel

Transcript (.txt)
Get:

Per-criterion score

Keyword match report

Length analysis

JSON downloadable result
