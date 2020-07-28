# %% import dataframe from pickle file
import pandas as pd

df = pd.read_pickle("UK.pkl")

df.head()


# %% convert dataframe to invoice-based transactional format



# %% apply apriori algorithm to find frequent items and association rules



# %% count of frequent itemsets that have more then 1/2/3 items,
# and the frequent itemsets that has the most items



# %% top 10 lift association rules



# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot support vs lift
