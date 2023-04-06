run_api:
	uvicorn chirpID.api.fast:app --reload

run_st:
	streamlit run app.py
