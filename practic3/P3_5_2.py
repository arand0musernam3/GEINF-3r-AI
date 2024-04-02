# NAME: Aniol Juanola Vilalta (u1978893)
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401

le_owners = LabelEncoder()


def read_model_5():
    df = pd.read_csv("steam.csv")
    df = df.drop(columns=['appid', 'name', 'release_date', 'developer', 'publisher', 'categories', 'positive_ratings',
                          'negative_ratings', 'median_playtime', 'price', 'genres'])
    print("NaN values")
    print("==========")

    df = df.dropna()
    print()
    # converting the categorical variables into numerical
    df['owners'] = le_owners.fit_transform(df['owners'])
    return df


steamData = read_model_5()


print(steamData)

## now we need to scale the data for the distances to work properly

scaler = MinMaxScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(steamData))
scaled_features.columns = steamData.columns

## Performance of similar measures

from sklearn.cluster import HDBSCAN
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
#K = range(2, 15)
K = 1
for k in range(1,2):
    model5 = HDBSCAN()
    labels5 = model5.fit_predict(steamData)
    #auxMatrix = model5.transform(steamData)

    if not os.path.exists("output/ex5/" + str(k)):
        os.mkdir("output/ex5/" + str(k))

    #distortions.append(sum(np.min(cdist(steamData, model5.cluster_centers_,
    #                                    'euclidean'), axis=1)) / steamData.shape[0])
    #inertias.append(model5.inertia_)
    #
    #mapping1[k] = sum(np.min(cdist(steamData, model5.cluster_centers_,
    #                               'euclidean'), axis=1)) / steamData.shape[0]
    #mapping2[k] = model5.inertia_

    unique_labels5 = np.unique(labels5)
    print("k=", k)
    #count_labels5 = [0] * k
    #sum_labels5 = [0] * k
    #for i in labels5:
    #    count_labels5[i] = count_labels5[i] + 1

    #for i in auxMatrix:
    #    j = np.argmin(i)
    #    sum_labels5[j] = sum_labels5[j] + i[j]

    ## Cluster cardinality
    #plt.bar(unique_labels5, count_labels5)
    #plt.xlabel("Cluster")
    #plt.ylabel("Number of occurrences")
    #plt.title("Cluster cardinality")
    #plt.savefig("./output/ex5/" + str(k) + "/cardinality.png")
    #plt.clf()

    ## Cluster magnitude
    ### Sum of distances from all examples to the centroid of the cluster
    #plt.bar(unique_labels5, sum_labels5)
    #plt.xlabel("Cluster")
    #plt.ylabel("Sum of intracluster distances")
    #plt.title("Cluster magnitude")
    #plt.savefig("./output/ex5/" + str(k) + "/magnitude.png")
    #plt.clf()

    ## let's check whether these are on the same cluster or not:
    #auxArray2 = steamData['appid'].values
    #for key, value in steamData.items():
    #    auxArray = [0] * k
    #    for i in value:
    #        instance = np.where(auxArray2 == i)
    #        auxLabel = labels5[int(instance[0])]
    #        auxArray[auxLabel] = auxArray[auxLabel] + 1
    #
    #    plt.bar(unique_labels5, auxArray)
    #    plt.xlabel("Cluster")
    #    plt.ylabel("Number of instances")
    #    plt.title(key + " testing")
    #    plt.savefig("./output/ex5/" + str(k) + "/" + key + ".png")
    #    plt.clf()

## Elbow method
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.savefig("./output/ex5/elbowInertiaByK.png")

## It can be seen that the optimal number of clusters is at 5 or 6. Let's study them by taking a look at the
## generated graphs.
