# Campus Facility Reservation & Monitoring — Streamlit

Streamlit conversion of the uploaded notebook's reservation analytics workflow.

## Files
- `app.py` — interactive dashboard
- `requirements.txt` — Python dependencies
- `README.md` — setup guide

## Run locally
Use Python 3.10 or newer. In a terminal opened in this folder, run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually http://localhost:8501).

## Data
Upload `facility_reservations.csv` using the sidebar uploader. The app supports the notebook's `User_Category`/`User_Type` and `Lead_Time_Days`/`Lead_Time` column variants. Without an uploaded file, it uses synthetic demonstration data, not real campus records.

The app includes dataset inspection, data cleaning summary and CSV export, exploratory analysis, decision-tree evaluation, an interactive prediction form, functional requirements, and a use-case overview. Post-event fields are excluded from model predictors to reduce target leakage. Predictions are estimates; messaging and live booking integrations are not implemented.
