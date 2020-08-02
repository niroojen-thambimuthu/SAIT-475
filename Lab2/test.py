# dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
#        "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
#        "area": [8.516, 17.10, 3.286, 9.597, 1.221],
#        "population": [200.4, 143.5, 1252, 1357, 52.98] }

# import pandas as pd
# brics = pd.DataFrame(dict)
# print(brics)


# import seaborn as sns
# import matplotlib.pyplot as plt
# tips = sns.load_dataset("tips")
# # ax = sns.scatterplot(x="total_bill", y="tip", data=tips)
# ax = sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)
# plt.title("Some Title")
# plt.xlabel("Some xaxis label")
# plt.ylabel("Some yaxis label")
# plt.show()

# %%
import pandas as pd

# s = pd.Series([1, 10e1, 10e2, 10e3, 10e4]

d = pd.DataFrame({
    "col1": range(100),
    "col2": [i*2 for i in range(100)],
    "col3": [i**2 for i in range(100)],
})

# %%
d.head()

# %%
d.columns

# %%
d.index

# %%
d["col1"]


# %%
d

# %%
d[53:56]

# %%
d.iloc[55]


# %%
d.loc[55]


# %%
# d[55]

# %%
d.index = [f"a{i}" for i in range(100)]
d.loc["a33"]

# %%
d[d["col3"] < 400]

# %%
d[(d["col3"] < 400) & (d["col1"] > 3)]


# %%
d[(d["col3"] > 400) | (d["col1"] < 3)]

# %%
d[~(d["col3"] < 400)]


# %%
import numpy as np

d.loc["a0", "col1"] = np.nan
d.loc["a1", "col2"] = np.nan
d.loc["a2", "col3"] = np.nan 

# %%
d

# %%
d.dropna()

# %%
d.fillna(-100)


# %%
d.isna()


# %%
d.describe()


# %%
d.mean(axis=1)


# %%
d.apply(lambda col: col.sum() / col.count())




# %%
d.sum()
d.count()

# %%

a = d["col3"].astype("str")

# %%
a = a.str.rstrip(".0") 



# %%
a.str.zfill(6)



# %%
s = pd.Series(np.random.randint(0, 7, size=100)) 
s.value_counts()


# %%
s.value_counts()[:3]

# %%
s.value_counts(ascending=True)[:3]

# %%
df = pd.DataFrame({
    'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2': [2, 1, 9, 8, 7, 4],
    'col3': [0, 1, 9, 4, 2, 3],
})
df
df.sort_values(by=['col1'])

# %%
df.sort_values(by=['col1', 'col2'])

# %%
df.sort_values(by=['col1', 'col2'], ascending=False)

# %%
df = pd.DataFrame(
    {
        "A": [
            "foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"
        ],
        "B": ["x", "x", "y", "z", "y", "y", "x", "z",],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

df

# %%
df.groupby('A').sum()

# %%
df.groupby(['A', 'B']).sum()

# %%
for key, group_df in df.groupby('A'):
    print(f"{key}: {type(group_df)}")
    group_df.head()
    print()


# %%
