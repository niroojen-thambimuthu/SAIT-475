# %% import dataframe from pickle file
import pandas as pd

df = pd.read_pickle("UK.pkl")

df.head()


# %% convert dataframe to invoice-based transactional format
dataset = df.groupby(["InvoiceNo"]).apply(lambda f: f["Description"].tolist())


# %% apply apriori algorithm to find frequent items and association rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)


# %% count of frequent itemsets that have more then 1/2/3 items,
# and the frequent itemsets that has the most items
length = frequent_itemsets["itemsets"].apply(len)

print(f"{(length > 1).sum()} itemsets have more than 1 item")
print(f"{(length > 2).sum()} itemsets have more than 2 item")
print(f"{(length > 3).sum()} itemsets have more than 3 item")

frequent_itemsets[length == length.max()]

# %% top 10 lift association rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, min_threshold=0.6)
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


# %%
