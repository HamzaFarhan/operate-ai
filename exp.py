import pandas as pd


path = "/private/var/folders/34/rqmrdz1n7rj_r1f2x_hsjkkr0000gn/T/ai_cfo_report_20250505_220115.xlsx"
pd.read_excel(path).to_dict(orient="records")
