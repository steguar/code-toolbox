# CODE Interactive Platform (Streamlit)

This repository can be deployed as a web app via **Streamlit**.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The web app exposes three modules: Network generation (RHBM), Epidemic model, and Opinion dynamics.
- Network generation uses the RHBM scripts under `network/geometric_block_model/src/rhbm/` via `subprocess`.
- For the epidemic and opinion modules, you may upload `nodes.txt` and `edges.txt` directly from the sidebar.

