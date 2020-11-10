# %% read dataframe from part1
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_pickle("sqf.pkl")


# %% select some yes/no columns to convert into a dataframe of boolean values
pfs = [col for col in df.columns if col.startswith("pf_")]

armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[pfs + armed]
x = x == "YES"


# %% create a new column to represent whether a person is armed
x["armed"] = (
    x["contrabn"]
    | x["pistol"]
    | x["riflshot"]
    | x["asltweap"]
    | x["knifcuti"]
    | x["machgun"]
    | x["othrweap"]
)

# %% select some categorical columns and do one hot encoding
for val in df["race"].unique():
    x[f"race_{val}"] = df["race"] == val


for val in df["city"].value_counts().index:
    x[f"city_{val}"] = df["city"] == val


for val in df["sex"].value_counts().index:
    x[f"sex_{val}"] = df["sex"] == val

# %% apply frequent itemsets mining, make sure you play around of the support level
# lowered min support for more frequent itemsets
frequent_itemsets = apriori(x, min_support=0.10, use_colnames=True)
frequent_itemsets.sort_values(by="support", ascending=False)

# %% count of frequent itemsets that have more then 1/2/3 items,
# and the frequent itemsets that has the most items
length = frequent_itemsets["itemsets"].apply(len)

# print(f"{(length > 1).sum()} itemsets have more than 1 item")
# print(f"{(length > 2).sum()} itemsets have more than 2 item")
# print(f"{(length > 3).sum()} itemsets have more than 3 item")

frequent_itemsets[length == length.max()]


# %% apply association rules mining 
rules = association_rules(frequent_itemsets, min_threshold=0.93)
rules

# %%
rules.sort_values(["lift"], ascending=False).head(10)

# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot support vs lift
sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("Support vs lift")


# %% sort rules by confidence and select rules within "armed" in it
rules.sort_values("confidence", ascending=False)[
    rules.apply(
        lambda r: "armed" in r["antecedents"]
        or "armed" in r["consequents"],
        axis=1,
    )
]

# %%
