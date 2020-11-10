# %% read csv file
import pandas as pd

df = pd.read_csv("2012.csv") ## 532911 rows x 112 columns

# df = df[(df["age"] <= 30) & (df["age"] >= 17)]
# df = df['age'].unique()
# df["age"].value_counts().head(20)

# df

# for col in df.columns: 
#     print(col)

# dropna here gives 532805 rows x 112 columns



# %% make sure numeric columns have numbers
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
## dropna here gives 516986 rows x 112 columns

# %% Drop duplicates
df = df.drop_duplicates()

# %% make datetime column
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime

def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(
    lambda r: make_datetime(r["datestop"], r["timestop"]), 
    axis=1
)


# %% convert all value to label in the dataframe, remove rows that cannot be mapped
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()


# %% convert xcoord and ycoord to (lon, lat)
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)

# %% convert height in feet/inch to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% remove outlier
df = df[(df["age"] <= 70) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]
df = df.drop(df[df['sex']=="UNKNOWN"].index) 
df = df[(df["perobs"] <= 300)] ## rows at 492251 x 79 columns


# %% delete columns that are not needed
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",        
        
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop 
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %%
# ## Remove unknown outliers in sex column
# df = df.drop(df[df['sex']=="UNKNOWN"].index) ## rows at 492265 x 79 columns
# ## period of observation filters
# df = df[(df["perobs"] <= 300)]

## statistics: age, sex, arstmade, perobs - Replace as we go
# # %% Most Statistics
# df["perobs"].describe(include='all')
# # %%
# df["perobs"].value_counts().head(20)
# # %% Mode
# df["perobs"].mode()
# # %% Median
# df["perobs"].median()
# # %% Variance
# df["perobs"].var()
# # %%
# df["perobs"].max()



# %%
import folium 
import seaborn as sns
import matplotlib.pyplot as plt

## month number to month bug correction
# ax = sns.countplot(df["datetime"].dt.month)
# ax.set(xlabel='Month of Year')


# %%
# ax = sns.countplot(df["datetime"].dt.weekday)
# # The day of the week with Monday=0, Sunday=6. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.weekday.html
# ax.set_xticklabels(
#     ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
# )
# ax.set(xlabel='Day of Week')


# # %%
# ax = sns.countplot(df["datetime"].dt.hour)
# ax.set(xlabel='Hour of Day')


# %% Race
# sns.set_palette("GnBu_d")
# ax = sns.countplot(data=df,x="race", order=df["race"].value_counts().iloc[:5].index)
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=10, fontsize=9
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("TOP 5 SQF RACE COUNT")
# plt.xlabel("RACE")
# plt.ylabel("COUNT")


# # %% sex
# sns.set_palette([ "#3498db","#9b59b6"])
# ax = sns.countplot(data=df,x="sex")
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=0, fontsize=9
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("SQF SEX COUNT")
# plt.xlabel("SEX")
# plt.ylabel("COUNT")


# # %% Age
# sns.set_palette([ "#8B0000"])
# ax = sns.distplot(df["age"], bins=10)
# plt.title("SQF AGE DISTRIBUTION")
# plt.xlabel("AGE")


# # %% city
# sns.set_palette("cubehelix", 8)
# ax = sns.countplot(data=df,x="city")
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=0, fontsize=10
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("SQF COUNT PER CITY")
# plt.xlabel("CITY")
# plt.ylabel("COUNT")


# # %% Arrest Made, assuming weapon or contraband is found
# sns.set_palette([ "#FF0000","#008000"])
# ax = sns.countplot(data=df,x="arstmade")
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=0, fontsize=12
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("SQF ARREST MADE COUNT")
# plt.xlabel("WAS ARREST MADE?")
# plt.ylabel("COUNT")



# # %% perobs
# sns.set_palette("GnBu_d")
# ax = sns.countplot(data=df,x="perobs", order=df["perobs"].value_counts().iloc[:5].index)
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=0, fontsize=12
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("TOP 5 SQF PERIOD OF OBSERVATION")
# plt.xlabel("OBSERVATION TIME IN MINUTES")
# plt.ylabel("COUNT")


# ################################################################


# # %% TOP 5 SQF RACE COUNT BY CITY
# sns.set_palette("cubehelix", 8)
# ax = sns.countplot(df["race"], hue=df["city"], order=df["race"].value_counts().iloc[:5].index)
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=10, fontsize=9
# )
# plt.title("TOP 5 SQF RACE COUNT BY CITY")
# plt.xlabel("RACE")
# plt.ylabel("COUNT")




# # %% SQF ARREST MADE COUNT BY SEX
# sns.set_palette(["#3498db","#9b59b6"])
# ax = sns.countplot(df["arstmade"], hue=df["sex"])
# ax.set_xticklabels(
#     ax.get_xticklabels(), rotation=0, fontsize=12
# )
# for p in ax.patches:
#     percentage = p.get_height() * 100 / df.shape[0]
#     txt = f"{percentage:.1f}%"
#     x_loc = p.get_x()
#     y_loc = p.get_y() + p.get_height()
#     ax.text(x_loc, y_loc, txt)
# plt.title("SQF ARREST MADE COUNT BY SEX")
# plt.xlabel("WAS ARREST MADE?")
# plt.ylabel("COUNT")

###############################################################

# %%
# nyc = (40.730610, -73.935242)

# m = folium.Map(location=nyc)


# # %%
# for coord in df.loc[df["detailcm"]=="CRIMINAL POSSESION OF CONTROLLED SUBSTANCE", "coord"]:
#     folium.CircleMarker(
#         location=(coord[1], coord[0]), radius=1, color="red"
#     ).add_to(m)

# m


# # %%
# for coord in df.loc[df["detailcm"]=="CRIMINAL SALE OF CONTROLLED SUBSTANCE", "coord"]:
#     folium.CircleMarker(
#         location=(coord[1], coord[0]), radius=1, color="yellow"
#     ).add_to(m)

# m





#########################################################
## Reason for SQF vs pf_

# %%
forces = [col for col in df.columns if col.startswith("pf_")] ## Physical force used by officer
reason = [col for col in df.columns if col.startswith("cs_") or col.startswith("rf_")] ## Reason for Stop and Reason for frisk

subset = df[forces + reason]
subset = (subset == "YES").astype(int)

f,ax = plt.subplots(figsize=(14, 10))
sns.heatmap(subset.corr(), linewidths=.2, fmt= '.1f',ax=ax, vmin=0.0, vmax=1.0, annot=True, cmap="Greens")
plt.title("CORRELATION - SQF REASON VS PHYSICAL FORCE BY OFFICERS")
plt.xlabel("SQF REASON AND TYPE OF FORCE")
plt.ylabel("SQF REASON AND TYPE OF FORCE")



# %%
df.to_pickle("sqf.pkl")