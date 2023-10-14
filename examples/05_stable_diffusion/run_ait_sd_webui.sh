#!/bin/bash
uvicorn server:app --host 0.0.0.0 --port 5000 --workers 1 &
streamlit run ait_sd_app_client.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
