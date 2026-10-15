from pathlib import Path
import pandas as pd

path = Path(__file__).parent / "evolution1.xlsx"
df = pd.read_excel(path, sheet_name="NUTS3")
print(df)
