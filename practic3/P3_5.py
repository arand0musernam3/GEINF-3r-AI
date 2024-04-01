# NAME: Aniol Juanola Vilalta (u1978893)
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401

le_name = LabelEncoder()
le_developer = LabelEncoder()
le_publisher = LabelEncoder()
le_categories = LabelEncoder()
le_genres = LabelEncoder()
le_owners = LabelEncoder()


def read_model():
    df = pd.read_csv("steam.csv")
    print("NaN values")
    print("==========")
    print(df.isna().any())  # Developer and Publisher have NaN values
    print("==========")
    print()
    df['developer'] = df['developer'].fillna(
        "THIS_IS_A_NAN_VALUE_REPLACE_ME")  # Placeholder for future reference when dealing with NaNs
    df['publisher'] = df['publisher'].fillna("THIS_IS_A_NAN_VALUE_REPLACE_ME")
    df['release_date'] = pd.to_datetime(df['release_date']).astype('int64')
    # converting the categorical variables into numerical
    df['name'] = le_name.fit_transform(df['name'])
    df['developer'] = le_developer.fit_transform(df['developer'])
    df['publisher'] = le_publisher.fit_transform(df['publisher'])
    df['categories'] = le_categories.fit_transform(df['categories'])
    df['genres'] = le_genres.fit_transform(df['genres'])
    df['owners'] = le_owners.fit_transform(df['owners'])
    return df


steamData = read_model()
steamTagData = pd.read_csv('steamspy_tag_data.csv')

## We need to combine both into one big dataset. Before, we'll get rid of unnecessary variables which are not correlated
## at all.


correlation = steamData.corr()  # https://i.imgur.com/x85MUWZ.png
print(correlation)

## It can be seen that the name doesn't give any information at all, as it practically is the same as an AppId. The
## release_date does correlate with the appid, as those are given out in order, so the smaller the number is the older
## a game is. As we could see in the previous exercise, developer and publisher are most of the time the same so their
## correlation level is high as well. A similar thing happens with positive and negative ratings, whose values have to
## add up to 100% and are most of the time opposite. Lastly, the average and median playtime are closely related
## because those values might be close on most polled games.

## The most interesting variables to us are 'appid' (to match this dataframe with the other one', the 'genres' and
## 'categories' fields which we'll obtain from the other dataframe), the amount of positive and negative ratings,
## the average playtime and the price. Let's add the other attributes from the other dataset.

print(steamData.shape)
print(steamTagData.shape)

mergedData = pd.merge(steamData, steamTagData, on=["appid"])
mergedData = mergedData.drop(columns=["name", "release_date", "developer", "publisher", "required_age", "categories",
                                      "genres", "achievements", "median_playtime", "owners"])
print(mergedData)

## now we need to scale the data for the distances to work properly

mergedData = mergedData.dropna()  # we might lose some data, but we've got more than enough to get a decent study

scaler = MinMaxScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(mergedData))
scaled_features.columns = mergedData.columns


for k in range(2,50):
    model5 = KMeans(n_clusters=k)
    labels5 = model5.fit_predict(mergedData)

    ## https://developers.google.com/machine-learning/clustering/interpret

    ## Cluster cardinality
    unique_labels5 = np.unique(labels5)
    print("k=", k)
    count_labels5 = [0] * k
    sum_labels5 = [0] * k
    auxMatrix = model5.transform(mergedData)
    for i in labels5:
        count_labels5[i] = count_labels5[i] + 1

    for i in auxMatrix:
        j = np.argmin(i)
    sum_labels5[j] = sum_labels5[j] + i[j]

    plt.bar(unique_labels5, count_labels5)
    plt.xlabel("Cluster")
    plt.ylabel("Number of occurrences")
    plt.title("Cluster cardinality")
    plt.savefig("./output/ex5/" + str(k) + "-cardinality.pdf")
    plt.clf()

    ## Cluster magnitude
    ### Sum of distances from all examples to the centroid of the cluster
    plt.bar(unique_labels5, sum_labels5)
    plt.xlabel("Cluster")
    plt.ylabel("Sum of intracluster distances")
    plt.title("Cluster magnitude")
    plt.savefig("./output/ex5/" + str(k) + "-magnitude.pdf")
    plt.clf()

    ## Performance of similar measures
    ## KNOWN APPIDS (I created this list based on my personal experiences of the games and genres that I have played)
    known_appids = {
        'shooter': [
            2620,
            2630,
            2640,
            7940,
            10090,
            42700,
            202970,
            209650,
            214630,
            24960,
            681350,
            10,
            80,
            240,
            730,
            273110,
            227940,
            306950
        ],
        'strategy': [
            105450,
            221380,
            362740,
            3900,
            3910,
            3990,
            8800,
            8930,
            16810,
            65980,
            289070,
            200510,
            268500,
            266840,
            323190
        ],
        'car/truck simulators/games': [
            244210,
            805550,
            339790,
            365960,
            431600,
            256330,
            354160,
            458770,
            621830,
            310560,
            421020,
            690790,
            17430,
            24870,
            47870,
            7200,
            11020,
            228760,
            232910,
            243360,
            375900,
            600720
        ],
        'singleplayer': [
            400,
            620,
            49520,
            261640,
            729040,
            683320,
            48000,
            304430,
            264710,
            848450,
            391540,
            50,
            70,
            130,
            220,
            280,
            320,
            340,
            360,
            380,
            420,
            17410,
            319630,
            532210,
            554620
        ]
    }

    ## let's check whether these are on the same cluster or not:
    auxArray2 = mergedData['appid'].values
    for key, value in known_appids.items():
        auxArray = [0] * k
        for i in value:
            instance = np.where(auxArray2 == i)
            auxLabel = labels5[int(instance[0])]
            auxArray[auxLabel] = auxArray[auxLabel] + 1

        plt.bar(unique_labels5, auxArray)
        plt.xlabel("Cluster")
        plt.ylabel("Number of instances")
        plt.title(key + " testing")
        plt.savefig("./output/ex5/" + str(k) + "-" + key + ".pdf")
        plt.clf()

## Optimum number of clusters
## TODO Amb HDBSCAN no n'hi ha, però amb Kmeans si!!
## TODO mirar els centroides!
