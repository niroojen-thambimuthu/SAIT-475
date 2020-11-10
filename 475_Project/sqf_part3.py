# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% pick a criminal code you want, probably not the one with high counts, in case the folium map hangs
df_assault = df[df["detailcm"] == "CRIMINAL SALE OF CONTROLLED SUBSTANCE"]

df_assault["lat"] = df["coord"].apply(lambda val: val[1])
df_assault["lon"] = df["coord"].apply(lambda val: val[0])


# %% run hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10

for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_assault[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_assault[["lat", "lon"]], y) ###
    labels[k] = y

# %%
best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
best_k

# %%
silhouette_scores

# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
ax.get_figure().savefig("trend.png", bbox_inches="tight", dpi=400)
plt.title("SILHOUETTE SCORES vs NUMBER OF CLUSTERS")
plt.xlabel("NUMBER OF CLUSTERS")
plt.ylabel("SILHOUETTE SCORE")

# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_assault["label"] = labels[best_k]

colors = sns.color_palette("hls", best_k).as_hex()

for row in tqdm(df_assault[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

# %%
best_k

# %%
m

#############################################################################

# %% find reason for stop columns
css = [col for col in df.columns if col.startswith("cs_")]
x = df_assault[css] == "YES"

# %% run KMeans clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10
x["lat"] = df_assault["lat"]
x["lon"] = df_assault["lon"]

for k in tqdm(range(num_city, num_pct, step)):
    km = KMeans(n_clusters=k)
    y = km.fit_predict(x)
    silhouette_scores[k] = silhouette_score(x, y)
    labels[k] = y


# %%
best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
best_k

# %%
silhouette_scores


# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)


# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
x["label"] = labels[best_k]

colors = sns.color_palette("hls", best_k).as_hex()

for row in tqdm(x[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m


# # %% find reason for stop columns
# css = [col for col in df.columns if col.startswith("cs_")]


# # %% run dbscan on reason for stop columns
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# from tqdm import tqdm

# c = DBSCAN()
# x = df_assault[css] == "YES"
# y = c.fit_predict(x)
# print(silhouette_score(x, y))


# # %% visualize the new clustering label on a map
# import numpy as np

# nyc = (40.730610, -73.935242)
# m = folium.Map(location=nyc)
# df_assault["label"] = y
# colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
# for row in tqdm(df_assault[["lat", "lon", "label"]].to_dict("records")):
#     folium.CircleMarker(
#         location=(row["lat"], row["lon"]),
#         radius=0.1,
#         color=colors[row["label"]],
#         alpha=0.3,
#     ).add_to(m)
# m


# # %% pick some of the labels to see if there's any location wise insight
# import numpy as np

# nyc = (40.730610, -73.935242)
# m = folium.Map(location=nyc)
# df_assault["label"] = y
# colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
# for row in tqdm(
#     df_assault.loc[df_assault["label"] == y[0], ["lat", "lon", "label"]].to_dict(
#         "records"
#     )
# ):
#     folium.CircleMarker(
#         location=(row["lat"], row["lon"]),
#         radius=0.1,
#         color=colors[row["label"]],
#         alpha=0.3,
#     ).add_to(m)
# m


# # %% pick some of the labels to see if there's any location wise insight
# import numpy as np

# nyc = (40.730610, -73.935242)
# m = folium.Map(location=nyc)
# df_assault["label"] = y
# colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
# for row in tqdm(
#     df_assault.loc[df_assault["label"] == y[1], ["lat", "lon", "label"]].to_dict(
#         "records"
#     )
# ):
#     folium.CircleMarker(
#         location=(row["lat"], row["lon"]),
#         radius=0.1,
#         color=colors[row["label"]],
#         alpha=0.3,
#     ).add_to(m)
# m


# # %% pick some of the labels to see if there's any location wise insight
# import numpy as np

# nyc = (40.730610, -73.935242)
# m = folium.Map(location=nyc)
# df_assault["label"] = y
# colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
# for row in tqdm(
#     df_assault.loc[df_assault["label"] == 63, ["lat", "lon", "label"]].to_dict(
#         "records"
#     )
# ):
#     folium.CircleMarker(
#         location=(row["lat"], row["lon"]),
#         radius=0.1,
#         color=colors[row["label"]],
#         alpha=0.3,
#     ).add_to(m)
# m


# %%
