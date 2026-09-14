# Schulwegplan Berlin — Mitte

Interactive school-route-safety map for Berlin-Mitte primary schools (catchments,
walking isochrones, route hierarchy, and pedestrian-safety context layers), built
for a Master's thesis on school-route planning.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deployed on Streamlit Community Cloud

Set a `PUBLIC_APP_URL` secret (Settings → Secrets) to this app's own
`https://...streamlit.app` address so the in-app "Open on phone" QR code and
link point at the live deployment instead of a local network address:

```toml
PUBLIC_APP_URL = "https://your-app-name.streamlit.app"
```

## Data

`data/interim/` contains only the pre-processed GeoPackages this app reads at
runtime (isochrones, catchments, route hierarchy, context-safety layers, OSM
pedestrian network extract). The full thesis data-processing pipeline
(notebooks, raw sources, intermediate outputs) lives in the main project
repository and is not included here.
