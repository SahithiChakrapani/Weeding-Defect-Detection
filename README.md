# Weeding Defect Detector — Streamlit App

A web UI for the HARP/INC test-sheet defect detection pipeline. Upload one
image, see the annotated result with bounding boxes around defective and
missing letters, and review a per-defect breakdown.

## Files

- **app.py** — Streamlit front-end (UI only)
- **pipeline.py** — Detection algorithm, ported verbatim from the notebook
- **requirements.txt** — Python dependencies
- **README.md** — This file

## Setup (one-time)

Make sure you have Python 3.10+ installed.

```bash
# Optional but recommended: create a virtual environment
python -m venv venv
# Activate it
#   Windows:    venv\Scripts\activate
#   Mac/Linux:  source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the app locally

```bash
python -m streamlit run app.py
```

Your browser opens automatically at **http://localhost:8501**.

## How to use

1. Click **Browse files** and upload a HARP/INC test sheet image (JPG/PNG).
2. Click **Run Detection**.
3. Review the annotated image, summary tally, and per-defect breakdown.
4. Optionally expand **Reference Templates Used** to see how the algorithm
   built its "ideal" reference for comparison.

## Scope

This app is **specifically for HARP/INC test sheets** — repeating-letter
quality-control sheets with the standard layout (HARP letters on the left,
INC group on the right, multiple rows of identical content).

It will not work on:
- Generic logos or single-instance graphics
- Test sheets with different layouts
- Photos with poor lighting or extreme perspective tilt

The image must show:
- A bright sheet on a darker background (for auto-cropping to work)
- Multiple rows of letters (so a reference template can be built from medians)
- Clear, well-lit letters with reasonable contrast

## Deploy to Streamlit Community Cloud (free public hosting)

1. Push these files to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click **New app**, point it at your repo, set Main file path to `app.py`,
   click **Deploy**.
4. After ~3–5 minutes you'll get a URL like
   `https://your-app-name.streamlit.app`. Share that URL with anyone.
