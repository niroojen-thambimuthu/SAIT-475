# %%
# data
from sklearn import datasets
iris = datasets.load_iris()
# X, y = iris.data[:,:2], iris.target
# split into training/testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, random_state=123
)
# call learner
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# metrics
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, y_pred)}")

# %%
