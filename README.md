# CV / Resume Ranker — Screening Console
## https://cvranker.vercel.app
Rank a batch of CVs (PDFs) against a job description, in the browser.

Paste a job brief, drop in some PDFs, and get a ranked shortlist with a
relevance score, a grade band, and which required skills each candidate does
and doesn't mention.

**Nothing is uploaded.** PDF parsing and scoring both run in the visitor's
browser — the files never touch a server, because there isn't one.

## How it works

- **PDF text extraction** — [pdf.js](https://mozilla.github.io/pdf.js/) reads
  each file locally and rebuilds line breaks (the candidate-name heuristic
  depends on real lines, not one flat string).
- **Scoring** — [transformers.js](https://huggingface.co/docs/transformers.js)
  runs `all-MiniLM-L6-v2` in-page (WebGPU where available, WASM otherwise). The
  job brief and each CV are embedded, then ranked by cosine similarity, scaled
  to a 0–100% relevance score.
- **Skill matching** — plain case-insensitive substring check per required
  skill, shown as matched/missing chips.

No LLM is involved. This is a small **embedding model** — the right tool for
"how semantically similar is this CV to this job", fast enough to run on a
phone, and free.

The model is ~23MB quantised. It downloads once on the first visit and is
cached by the browser after that; subsequent runs work offline.

## Running it locally

It's a static site — no build step, no dependencies. Serve the folder:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>. (Opening `index.html` via `file://` will
not work — ES modules and `fetch` need a real origin.)

## Deploying

Push to GitHub and import the repo on [Vercel](https://vercel.com). Framework
preset **Other**, no build command, output directory `.` — `vercel.json`
handles the rest.

## Layout

```
index.html      markup
style.css       design tokens + layout
app.js          PDF extraction, embedding, ranking, demo mode
samples/        six synthetic CVs used by "Try it with sample data"
legacy-flask/   the original Flask + sentence-transformers backend
```

## Sample data

The **Try it with sample data** button loads a Senior Backend Engineer brief
and six synthetic CVs from `samples/`, then runs a real screening. The
candidates are invented and deliberately span the grade bands — a close match,
a couple of adjacent-stack engineers, and some clear mismatches — so the
ranking visibly does something.

Sample CVs are generated, not real. Do not commit real candidate CVs to this
repo: `CVs/` and `uploads/` are gitignored for that reason.

## Notes & limits

- Only text-based PDFs work. Scanned or image-only PDFs have no extractable
  text and show as **Unreadable File** — handling those needs OCR, which isn't
  included.
- Candidate name detection is a heuristic (first plausible line, else the
  filename). Double-check names on the results page.
- Only the first 8,000 characters of each CV are scored, and the model itself
  reads roughly the first 256 tokens — enough for a summary and skills
  section, not a full career history.
- Results are not saved. Refreshing clears the shortlist.
- Treat the score as a fast first-pass shortlist, **not** a hiring decision.

## The original Flask version

`legacy-flask/` holds the previous implementation: the same scoring logic in
Python with `sentence-transformers`, serving uploads from disk. It's kept for
reference. It can't run on Vercel — PyTorch is far larger than the serverless
bundle limit — so if you want the Python version hosted, use a container host
(Hugging Face Spaces, Render, Fly.io) instead.
