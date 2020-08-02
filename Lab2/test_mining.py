# %%
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'],
] 

dataset

# %%
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)


# %%
frequent_itemsets

# %%
from mlxtend.frequent_patterns import association_rules 

rules = association_rules(frequent_itemsets, min_threshold=0.1) 
rules

# %%
import seaborn as sns
import matplotlib.pyplot as plt 
ax = sns.scatterplot(
    x="support", y="confidence", alpha=0.5, data=rules
)
plt.show()

# %%
length = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["length"] = length 
frequent_itemsets

# %%
rules.sort_values("confidence", ascending=False).head(3)

# %%
