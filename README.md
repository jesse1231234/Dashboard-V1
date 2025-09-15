# Course Analytics Streamlit Dashboard (Test Pack)

This folder contains a **self-contained** Streamlit app and **sample CSVs** so you can test the dashboard immediately.

## Files
- `app.py` — the integrated, secure Streamlit app (no external modules required)
- `requirements.txt` — dependencies
- `.gitignore` — prevents committing secrets
- `.streamlit/secrets.example.toml` — template for secrets (safe to commit)
- `sample_canvas_order.csv` — Canvas order demo
- `sample_echo.csv` — Echo360 demo
- `sample_gradebook.csv` — Gradebook demo

## Quick Start
1. (Optional but recommended) create a venv.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. In the sidebar:
   - For **Canvas Materials Order**, choose **Upload CSV** and upload `sample_canvas_order.csv`.
   - Upload `sample_echo.csv` under **Echo360 CSV**.
   - Upload `sample_gradebook.csv` under **Gradebook CSV**.
   The KPIs, tables, and charts should populate.

## Using the real API (secure)
- Create a **local** `.streamlit/secrets.toml` (do **not** commit it):
  ```toml
  [canvas]
  BASE_URL = "https://colostate.instructure.com"
  API_TOKEN = "YOUR_REAL_CANVAS_TOKEN"
  ```
- Or set env vars: `CANVAS_BASE_URL` and `CANVAS_API_TOKEN`.
- In the app sidebar, choose **Use API**, toggle **Use server secret**, enter a **Course ID**, and click **Fetch Canvas Order**.

> Tokens are never logged or cached by the app. You can always switch back to **Upload CSV** to avoid using an API key.

## Notes
- The app is flexible about column names and tries smart matches.
- The student-level table is de-identified (IDs 1..N).
- Charts require both the order + echo or order + gradebook to be present.
