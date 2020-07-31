# %% import pandas and read the csv file 
# modify the path if needed
import pandas as pd

df = pd.read_csv("DATA475_lab_rectangles_data.csv")
df["area"] = df["width"] * df["length"]  # Line 6

# %%
summary = [
    ("Total Count", df["area"].shape[0]),
    ("Total Area", df["area"].sum()),
    ("Average Area", df["area"].mean()),   # Line 12
    ("Maximum Area", df["area"].max()),    # Line 13
    ("Minimum Area", df["area"].min()),    # Line 14
]

for key, value in summary:
    print(f"{key}: {str(value)}")

# %%
d = dict(summary)    # Line 21
for key in d:
    d[key] = [d[key]]

pd.DataFrame(d).to_csv("summary.csv", index=False)


# %%
