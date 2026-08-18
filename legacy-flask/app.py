"""
CV / Resume Ranker — backend
-----------------------------
Flask app that:
  1. Serves the frontend (static/index.html, style.css, script.js)
  2. Accepts a job description + a batch of CV PDFs at POST /rank
  3. Extracts text from each PDF, embeds it with a local sentence-transformer
     model, scores it against the job requirements with cosine similarity,
     and returns a ranked JSON list.
  4. Serves the original uploaded CVs back at /uploads/<filename> so the
     frontend can link/preview them.

Everything runs 100% locally — no API key, no internet call at inference time
(the embedding model is downloaded once the first time you run this).
"""

import os
import re
import uuid
from pathlib import Path

import numpy as np
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors (no sklearn needed)."""
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB total upload cap

print("Loading local embedding model (first run downloads it, ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded. Ready.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath: Path) -> str:
    """Pull all text out of a PDF using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        print(f"Failed to read {filepath.name}: {e}")
    return "\n".join(text_parts)


def guess_candidate_name(text: str, fallback_filename: str) -> str:
    """
    Heuristic: the candidate's name is usually the first non-empty line of
    the resume, and it's short, has no digits/emails, and isn't a section
    heading like 'RESUME' or 'CURRICULUM VITAE'.
    Falls back to a cleaned-up version of the filename.
    """
    skip_words = {"resume", "cv", "curriculum vitae", "profile", "biodata"}
    for raw_line in text.splitlines()[:12]:
        line = raw_line.strip()
        if not line:
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        if line.lower() in skip_words:
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() or not w.isalpha() for w in words if w):
            if 2 <= len(line) <= 40:
                return line.title() if line.isupper() else line

    # fallback: clean the filename
    name = Path(fallback_filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\b(resume|cv|final|updated|copy)\b", "", name, flags=re.I).strip()
    return name.title() if name else fallback_filename


def extract_matched_skills(cv_text: str, required_skills: list[str]) -> tuple[list[str], list[str]]:
    text_lower = cv_text.lower()
    matched, missing = [], []
    for skill in required_skills:
        skill_clean = skill.strip()
        if not skill_clean:
            continue
        if skill_clean.lower() in text_lower:
            matched.append(skill_clean)
        else:
            missing.append(skill_clean)
    return matched, missing


def score_to_grade(score: float) -> str:
    if score >= 75:
        return "Excellent Match"
    elif score >= 55:
        return "Good Match"
    elif score >= 35:
        return "Average Match"
    else:
        return "Poor Match"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/rank", methods=["POST"])
def rank_cvs():
    job_category = request.form.get("job_category", "").strip()
    required_skills_raw = request.form.get("required_skills", "").strip()
    min_experience = request.form.get("min_experience", "").strip()
    education_level = request.form.get("education_level", "").strip()
    job_description = request.form.get("job_description", "").strip()

    if not job_category and not job_description:
        return jsonify({"error": "Please provide at least a job category or job description."}), 400

    files = request.files.getlist("cv_files")
    if not files:
        return jsonify({"error": "Please upload at least one CV (PDF)."}), 400

    required_skills = [s for s in required_skills_raw.split(",") if s.strip()]

    # Build the "query" text that represents what the HR is looking for
    job_query = " ".join(filter(None, [
        f"Job category: {job_category}." if job_category else "",
        f"Required skills: {required_skills_raw}." if required_skills_raw else "",
        f"Minimum experience: {min_experience} years." if min_experience else "",
        f"Education level required: {education_level}." if education_level else "",
        job_description,
    ]))
    job_embedding = model.encode(job_query, convert_to_numpy=True)

    results = []
    for f in files:
        if not f or not f.filename or not allowed_file(f.filename):
            continue

        unique_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        saved_path = UPLOAD_FOLDER / unique_name
        f.save(saved_path)

        cv_text = extract_text_from_pdf(saved_path)
        if not cv_text.strip():
            # Unreadable / scanned PDF with no extractable text
            results.append({
                "name": guess_candidate_name("", f.filename),
                "original_filename": f.filename,
                "stored_filename": unique_name,
                "score": 0,
                "grade": "Unreadable File",
                "matched_skills": [],
                "missing_skills": required_skills,
            })
            continue

        cv_embedding = model.encode(cv_text[:8000], convert_to_numpy=True)
        similarity = cosine_similarity(job_embedding, cv_embedding)
        score = round(max(0.0, min(1.0, similarity)) * 100, 1)

        matched, missing = extract_matched_skills(cv_text, required_skills)
        name = guess_candidate_name(cv_text, f.filename)

        results.append({
            "name": name,
            "original_filename": f.filename,
            "stored_filename": unique_name,
            "score": score,
            "grade": score_to_grade(score),
            "matched_skills": matched,
            "missing_skills": missing,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return jsonify({"results": results, "total": len(results)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
