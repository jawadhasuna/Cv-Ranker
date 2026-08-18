/**
 * CV / Resume Ranker — client-side engine
 * ---------------------------------------
 * Everything that used to live in app.py now runs in the visitor's browser:
 *   1. PDFs are read locally with pdf.js — no upload, no server.
 *   2. The job brief and each CV are embedded with all-MiniLM-L6-v2 running
 *      in-page via transformers.js (WebGPU where available, WASM otherwise).
 *   3. Candidates are ranked by cosine similarity, with the same keyword
 *      match and grade bands the Flask version used.
 *
 * No file ever leaves the machine it was opened on.
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.worker.min.mjs";

// We only ever load the model from the HF hub, never from a local /models path.
env.allowLocalModels = false;

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const MAX_CHARS = 8000; // mirrors cv_text[:8000] in the original backend

// --------------------------------------------------------------------------- //
// DOM
// --------------------------------------------------------------------------- //

const form = document.getElementById("rankForm");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("cv_files");
const fileCount = document.getElementById("fileCount");
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const errorMsg = document.getElementById("errorMsg");
const demoBtn = document.getElementById("demoBtn");

const resultsTitle = document.getElementById("resultsTitle");
const resultsSub = document.getElementById("resultsSub");
const resultsList = document.getElementById("resultsList");
const loadingState = document.getElementById("loadingState");
const loadingText = document.getElementById("loadingText");
const progressFill = document.getElementById("progressFill");
const legend = document.getElementById("legend");

// Blob URLs handed out for "open original CV" links, revoked between runs.
let activeObjectUrls = [];

// --------------------------------------------------------------------------- //
// Scoring helpers — direct ports of the Python originals
// --------------------------------------------------------------------------- //

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function scoreToGrade(score) {
  if (score >= 75) return "Excellent Match";
  if (score >= 55) return "Good Match";
  if (score >= 35) return "Average Match";
  return "Poor Match";
}

function titleCase(str) {
  return str.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

const SKIP_WORDS = new Set(["resume", "cv", "curriculum vitae", "profile", "biodata"]);

/**
 * The candidate's name is usually the first short, digit-free, capitalised
 * line of the resume. Falls back to a cleaned-up filename.
 */
function guessCandidateName(text, fallbackFilename) {
  const lines = text.split(/\r?\n/).slice(0, 12);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.includes("@") || /\d/.test(line)) continue;
    if (SKIP_WORDS.has(line.toLowerCase())) continue;

    const words = line.split(/\s+/).filter(Boolean);
    const capitalised = words.every((w) => {
      const first = w[0];
      const isAlphaWord = /^[A-Za-z]+$/.test(w);
      return !isAlphaWord || (first === first.toUpperCase() && first !== first.toLowerCase());
    });

    if (words.length >= 1 && words.length <= 4 && capitalised && line.length >= 2 && line.length <= 40) {
      return line === line.toUpperCase() ? titleCase(line) : line;
    }
  }

  let name = fallbackFilename.replace(/\.[^.]+$/, "");
  name = name.replace(/[_\-]+/g, " ");
  name = name.replace(/\b(resume|cv|final|updated|copy)\b/gi, "").trim();
  return name ? titleCase(name) : fallbackFilename;
}

function extractMatchedSkills(cvText, requiredSkills) {
  const lower = cvText.toLowerCase();
  const matched = [], missing = [];
  for (const raw of requiredSkills) {
    const skill = raw.trim();
    if (!skill) continue;
    (lower.includes(skill.toLowerCase()) ? matched : missing).push(skill);
  }
  return { matched, missing };
}

// --------------------------------------------------------------------------- //
// PDF text extraction
// --------------------------------------------------------------------------- //

async function extractTextFromPdf(file) {
  try {
    const buf = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    const parts = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();

      // Rebuild line breaks: pdf.js emits positioned spans, and the name
      // heuristic depends on real lines rather than one long string.
      let line = "";
      let lastY = null;
      for (const item of content.items) {
        if (item.str === undefined) continue;
        const y = item.transform ? item.transform[5] : null;
        if (lastY !== null && y !== null && Math.abs(y - lastY) > 2) {
          parts.push(line);
          line = "";
        }
        line += item.str;
        if (item.hasEOL) {
          parts.push(line);
          line = "";
        }
        lastY = y;
      }
      if (line) parts.push(line);
      parts.push("");
    }
    return parts.join("\n");
  } catch (err) {
    console.warn(`Failed to read ${file.name}:`, err);
    return "";
  }
}

// --------------------------------------------------------------------------- //
// Model
// --------------------------------------------------------------------------- //

let extractorPromise = null;

function getExtractor(onProgress) {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID, {
      dtype: "q8",
      progress_callback: onProgress,
    }).catch((err) => {
      extractorPromise = null; // let a later run retry a failed download
      throw err;
    });
  }
  return extractorPromise;
}

async function embed(extractor, text) {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return out.data;
}

// --------------------------------------------------------------------------- //
// UI
// --------------------------------------------------------------------------- //

function setProgress(pct) {
  if (progressFill) progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

const GRADE_CLASS = {
  "Excellent Match": "excellent",
  "Good Match": "good",
  "Average Match": "average",
  "Poor Match": "poor",
  "Unreadable File": "unreadable",
};

const gradeClass = (grade) => GRADE_CLASS[grade] || "unreadable";

function renderResults(results) {
  resultsList.innerHTML = "";

  if (!results.length) {
    resultsTitle.textContent = "No candidates scored";
    resultsSub.textContent = "No readable PDFs were found in that batch.";
    return;
  }

  resultsTitle.textContent = `${results.length} candidate${results.length > 1 ? "s" : ""} ranked`;
  resultsSub.textContent = "Sorted from strongest to weakest match against the role.";
  legend.hidden = false;

  for (const c of results) {
    const cls = gradeClass(c.grade);
    const row = document.createElement("div");
    row.className = `candidate-row ${cls}`;

    const skillsHtml = [
      ...c.matched_skills.map((s) => `<span class="chip matched">${escapeHtml(s)}</span>`),
      ...c.missing_skills.map((s) => `<span class="chip missing">${escapeHtml(s)}</span>`),
    ].join("");

    row.innerHTML = `
      <div class="rank-num">${String(c.rank).padStart(2, "0")}</div>
      <div class="candidate-info">
        <p class="candidate-name">${escapeHtml(c.name)}</p>
        <p class="candidate-file">
          <a href="${c.url}" target="_blank" rel="noopener">${escapeHtml(c.original_filename)}</a>
        </p>
        <div class="skills-line">${skillsHtml}</div>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-track">
          <div class="score-bar-fill ${cls}" style="width:${c.score}%"></div>
        </div>
        <span class="score-label">${c.score}% relevance</span>
      </div>
      <div class="grade-tag ${cls}">${escapeHtml(c.grade)}</div>
    `;
    resultsList.appendChild(row);
  }
}

function updateFileCount() {
  const n = fileInput.files.length;
  fileCount.textContent = n === 0 ? "No files selected" : `${n} file${n > 1 ? "s" : ""} selected`;
}

// --------------------------------------------------------------------------- //
// Main run
// --------------------------------------------------------------------------- //

async function runScreening() {
  errorMsg.textContent = "";

  const files = Array.from(fileInput.files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
  if (!files.length) {
    errorMsg.textContent = "Please choose at least one CV (PDF).";
    return;
  }

  const fd = new FormData(form);
  const jobCategory = (fd.get("job_category") || "").trim();
  const requiredSkillsRaw = (fd.get("required_skills") || "").trim();
  const minExperience = (fd.get("min_experience") || "").trim();
  const educationLevel = (fd.get("education_level") || "").trim();
  const jobDescription = (fd.get("job_description") || "").trim();

  if (!jobCategory && !jobDescription) {
    errorMsg.textContent = "Please provide at least a job category or a job description.";
    return;
  }

  const requiredSkills = requiredSkillsRaw.split(",").filter((s) => s.trim());

  const jobQuery = [
    jobCategory ? `Job category: ${jobCategory}.` : "",
    requiredSkillsRaw ? `Required skills: ${requiredSkillsRaw}.` : "",
    minExperience ? `Minimum experience: ${minExperience} years.` : "",
    educationLevel ? `Education level required: ${educationLevel}.` : "",
    jobDescription,
  ].filter(Boolean).join(" ");

  // Reset UI
  activeObjectUrls.forEach(URL.revokeObjectURL);
  activeObjectUrls = [];
  submitBtn.disabled = true;
  if (demoBtn) demoBtn.disabled = true;
  submitLabel.textContent = "Screening...";
  loadingState.hidden = false;
  resultsList.innerHTML = "";
  legend.hidden = true;
  resultsTitle.textContent = "Screening in progress";
  resultsSub.textContent = "";
  setProgress(0);

  try {
    // 1. Model — only the first visit actually downloads anything.
    loadingText.textContent = "Preparing the scoring model...";
    const seen = new Map();
    const extractor = await getExtractor((p) => {
      if (p.status === "progress" && p.file && p.total) {
        seen.set(p.file, { loaded: p.loaded, total: p.total });
        let loaded = 0, total = 0;
        for (const v of seen.values()) { loaded += v.loaded; total += v.total; }
        const pct = total ? Math.round((loaded / total) * 100) : 0;
        loadingText.textContent = `Downloading scoring model — ${pct}% (first visit only)`;
        setProgress(pct * 0.35);
      }
    });

    // 2. Job brief
    loadingText.textContent = "Reading the job brief...";
    setProgress(38);
    const jobVec = await embed(extractor, jobQuery);

    // 3. Each CV
    const results = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      loadingText.textContent = `Scoring ${i + 1} of ${files.length} — ${file.name}`;
      setProgress(40 + ((i + 1) / files.length) * 58);

      const url = URL.createObjectURL(file);
      activeObjectUrls.push(url);

      const text = await extractTextFromPdf(file);

      if (!text.trim()) {
        results.push({
          name: guessCandidateName("", file.name),
          original_filename: file.name,
          url,
          score: 0,
          grade: "Unreadable File",
          matched_skills: [],
          missing_skills: requiredSkills.map((s) => s.trim()),
        });
        continue;
      }

      const cvVec = await embed(extractor, text.slice(0, MAX_CHARS));
      const similarity = cosineSimilarity(jobVec, cvVec);
      const score = Math.round(Math.max(0, Math.min(1, similarity)) * 1000) / 10;
      const { matched, missing } = extractMatchedSkills(text, requiredSkills);

      results.push({
        name: guessCandidateName(text, file.name),
        original_filename: file.name,
        url,
        score,
        grade: scoreToGrade(score),
        matched_skills: matched,
        missing_skills: missing,
      });

      // Let the browser paint the progress bar between CVs.
      await new Promise((r) => setTimeout(r, 0));
    }

    results.sort((a, b) => b.score - a.score);
    results.forEach((r, i) => { r.rank = i + 1; });

    setProgress(100);
    renderResults(results);
  } catch (err) {
    console.error(err);
    errorMsg.textContent =
      "Could not finish screening: " + (err && err.message ? err.message : String(err));
    resultsTitle.textContent = "Results will appear here";
    resultsSub.textContent = "Run a screening to see candidates ranked by relevance.";
  } finally {
    loadingState.hidden = true;
    submitBtn.disabled = false;
    if (demoBtn) demoBtn.disabled = false;
    submitLabel.textContent = "Run Screening";
  }
}

// --------------------------------------------------------------------------- //
// Demo mode — pre-filled brief + synthetic CVs, so a visitor with no PDFs
// on hand can still see the thing actually work.
// --------------------------------------------------------------------------- //

const DEMO_SAMPLES = [
  "amara_okonkwo_cv.pdf",
  "daniel_reyes_cv.pdf",
  "priya_raghunathan_cv.pdf",
  "tomas_lindqvist_cv.pdf",
  "wei_chen_cv.pdf",
  "marcus_bello_cv.pdf",
];

const DEMO_BRIEF = {
  job_category: "Senior Backend Engineer",
  min_experience: "5",
  required_skills: "Python, Django, PostgreSQL, AWS, REST APIs",
  education_level: "Bachelor's Degree",
  job_description:
    "We are hiring a Senior Backend Engineer to own and scale the services behind our payments " +
    "platform. You will design and build REST APIs in Python and Django, model and tune relational " +
    "data in PostgreSQL, and run the whole thing on AWS. Day to day you will be shipping production " +
    "services, reviewing designs, tightening query performance, and improving our deployment and " +
    "observability story.\n\n" +
    "What we are looking for:\n" +
    "- 5+ years of professional backend engineering experience\n" +
    "- Strong Python, with production Django or Django REST Framework experience\n" +
    "- Confident relational data modelling and query optimisation in PostgreSQL\n" +
    "- Hands-on AWS experience (EC2, RDS, S3, Lambda) and containerised deployment\n" +
    "- Experience designing and versioning REST APIs used by other teams\n" +
    "- Comfortable with CI/CD, automated testing and being on-call for services you own\n\n" +
    "Nice to have: infrastructure as code (Terraform), async task queues such as Celery, " +
    "event streaming with Kafka, and mentoring more junior engineers.",
};

async function loadDemo() {
  errorMsg.textContent = "";
  if (demoBtn) {
    demoBtn.disabled = true;
    demoBtn.textContent = "Loading sample data...";
  }

  try {
    for (const [name, value] of Object.entries(DEMO_BRIEF)) {
      const el = form.elements[name];
      if (el) el.value = value;
    }

    const files = await Promise.all(
      DEMO_SAMPLES.map(async (name) => {
        const res = await fetch(`samples/${name}`);
        if (!res.ok) throw new Error(`could not load sample ${name} (${res.status})`);
        const blob = await res.blob();
        return new File([blob], name, { type: "application/pdf" });
      })
    );

    // Push the samples into the real file input so there is only one code path.
    const dt = new DataTransfer();
    files.forEach((f) => dt.items.add(f));
    fileInput.files = dt.files;
    updateFileCount();

    await runScreening();
  } catch (err) {
    console.error(err);
    errorMsg.textContent = "Could not load the sample CVs: " + (err.message || err);
  } finally {
    if (demoBtn) {
      demoBtn.disabled = false;
      demoBtn.textContent = "Try it with sample data";
    }
  }
}

// --------------------------------------------------------------------------- //
// Wiring
// --------------------------------------------------------------------------- //

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", updateFileCount);

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  const dropped = Array.from(e.dataTransfer.files).filter((f) =>
    f.name.toLowerCase().endsWith(".pdf")
  );
  if (!dropped.length) return;
  const dt = new DataTransfer();
  dropped.forEach((f) => dt.items.add(f));
  fileInput.files = dt.files;
  updateFileCount();
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runScreening();
});

if (demoBtn) demoBtn.addEventListener("click", loadDemo);
