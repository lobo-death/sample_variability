from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

data = load_iris()

dir(data)
df = pd.DataFrame(data.data, columns=["sepal_lenght", "sepal_width", "petal_lenght", "petal_width"])
df_y = pd.DataFrame(data.target, columns=["species"])

df["species"] = df_y

data.feature_names

data.target_names

mean = {"sepal length": [np.mean(data.data[:, 0])],
        "sepal width": [np.mean(data.data[:, 1])],
        "petal length": [np.mean(data.data[:, 2])],
        "petal width": [np.mean(data.data[:, 3])]}
mean_ = pd.DataFrame(mean)

resp_mean = {"mean": [mean_['sepal length'].values[0],
                      mean_['sepal width'].values[0],
                      mean_['petal length'].values[0],
                      mean_['petal width'].values[0]],
             "setosa": [np.mean(data.data[:, 0][0:50]),
                        np.mean(data.data[:, 1][0:50]),
                        np.mean(data.data[:, 2][0:50]),
                        np.mean(data.data[:, 3][0:50])],
             "versicolor": [np.mean(data.data[:, 0][50:100]),
                            np.mean(data.data[:, 1][50:100]),
                            np.mean(data.data[:, 2][50:100]),
                            np.mean(data.data[:, 3][50:100])],
             "virginica": [np.mean(data.data[:, 0][100:150]),
                           np.mean(data.data[:, 1][100:150]),
                           np.mean(data.data[:, 2][100:150]),
                           np.mean(data.data[:, 3][100:150])]}

df_mean = pd.DataFrame(resp_mean)

df_mean

sns.lineplot(data=df_mean)
plt.legend(["mean", "setosa", "versicolor", "virginica"])
plt.title("Average of the Values")
plt.xlabel("Values")
plt.ylabel("Quantity")
plt.savefig("images/mean_001.png")
plt.close()

std = {"sepal length": [np.std(data.data[:, 0])],
       "sepal width": [np.std(data.data[:, 1])],
       "petal length": [np.std(data.data[:, 2])],
       "petal width": [np.std(data.data[:, 3])]}
std_ = pd.DataFrame(std)

resp_std = {"std": [std_['sepal length'].values[0],
                    std_['sepal width'].values[0],
                    std_['petal length'].values[0],
                    std_['petal width'].values[0]],
            "setosa": [np.std(data.data[:, 0][0:50]),
                       np.std(data.data[:, 1][0:50]),
                       np.std(data.data[:, 2][0:50]),
                       np.std(data.data[:, 3][0:50])],
            "versicolor": [np.std(data.data[:, 0][50:100]),
                           np.std(data.data[:, 1][50:100]),
                           np.std(data.data[:, 2][50:100]),
                           np.std(data.data[:, 3][50:100])],
            "virginica": [np.std(data.data[:, 0][100:150]),
                          np.std(data.data[:, 1][100:150]),
                          np.std(data.data[:, 2][100:150]),
                          np.std(data.data[:, 3][100:150])]}
df_std = pd.DataFrame(resp_std)

df_std

sns.lineplot(data=df_std)
plt.legend(["std", "setosa", "versicolor", "virginica"])
plt.title("Standard Deviation of the Values")
plt.xlabel("Values")
plt.ylabel("Quantity")
plt.savefig("images/std_001.png")
plt.close()

x = data.data[:, 0:4]
y = data.target
sns.pairplot(df, hue="species")
plt.savefig("images/plot_001.png")
plt.close()
