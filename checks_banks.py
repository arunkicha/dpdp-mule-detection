import pandas as pd
e = pd.read_parquet("data/processed/edges_full.parquet")
C = [10,12,1,15,20,11,119,222,223,48309]
fb, tb = e.from_bank.astype(int), e.to_bank.astype(int)
for b in [70] + C:
    m = (fb == b) | (tb == b)
    print(b, int(m.sum()), int(e.is_laundering[m].sum()), round(float(e.is_laundering[m].mean()), 5))
print("overall rate", round(float(e.is_laundering.mean()), 5))