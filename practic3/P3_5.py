# NAME: Aniol Juanola Vilalta (u1978893)
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
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
scaledData = pd.DataFrame(scaler.fit_transform(mergedData))
scaledData.columns = mergedData.columns

## and we need to enhance the distances of the different categories so its impact is higher!
subset = scaledData.columns.drop(
    ['appid', 'english', 'positive_ratings', 'negative_ratings', 'price', 'average_playtime'])
multiplier = 10
scaledData.loc[:, subset] *= multiplier

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
    'car_simulator_games': [
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

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
labelsBestK = []
clustersBestK = []
K = range(2, 100)
bestK = -1
bestInertia = -1
for k in K:
    model5 = KMeans(n_clusters=k)
    labels5 = model5.fit_predict(scaledData)
    auxMatrix = model5.transform(scaledData)

    if not os.path.exists("output/ex5/" + str(k)):
        os.mkdir("output/ex5/" + str(k))

    distortions.append(sum(np.min(cdist(scaledData, model5.cluster_centers_,
                                        'euclidean'), axis=1)) / scaledData.shape[0])
    inertias.append(model5.inertia_)
    mapping1[k] = sum(np.min(cdist(scaledData, model5.cluster_centers_,
                                   'euclidean'), axis=1)) / scaledData.shape[0]
    mapping2[k] = model5.inertia_

    if bestK == -1 or bestInertia > model5.inertia_:
        bestK = k
        bestInertia = model5.inertia_
        labelsBestK = labels5
        clustersBestK = model5.cluster_centers_


    unique_labels5 = np.unique(labels5)
    print("k=", k)
    count_labels5 = [0] * k
    sum_labels5 = [0] * k
    for i in labels5:
        count_labels5[i] = count_labels5[i] + 1

    for i in auxMatrix:
        j = np.argmin(i)
        sum_labels5[j] = sum_labels5[j] + i[j]

    ## Cluster cardinality
    plt.bar(unique_labels5, count_labels5)
    plt.xlabel("Cluster")
    plt.ylabel("Number of occurrences")
    plt.title("Cluster cardinality")
    plt.savefig("./output/ex5/" + str(k) + "/cardinality.png")
    plt.clf()

    ## Cluster magnitude
    ### Sum of distances from all examples to the centroid of the cluster
    plt.bar(unique_labels5, sum_labels5)
    plt.xlabel("Cluster")
    plt.ylabel("Sum of intracluster distances")
    plt.title("Cluster magnitude")
    plt.savefig("./output/ex5/" + str(k) + "/magnitude.png")
    plt.clf()

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
        plt.savefig("./output/ex5/" + str(k) + "/" + key + ".png")
        plt.clf()

## Elbow method
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.savefig("./output/ex5/elbowInertiaByK.png")

## We will let inertia decide what the best clustering method is, as with so many columns it's very difficult to
## make a decision. We will then look at the Elbow graph to see if the choice matches the plot, and if it has landed
## on a value we will manually revise the lowest K value.

clustersBestK = pd.DataFrame(clustersBestK, columns=scaledData.columns)

# Define color scale for each column
color_scales = {
    'A': {1: 'red', 5: 'green'},
    'B': {6: 'red', 10: 'green'},
    'C': {100: 'red', 500: 'green'}
}


# Function to apply colors to the DataFrame
def apply_colors(value, color_scale):
    for threshold, color in color_scale.items():
        if value <= threshold:
            return color
    return 'white'  # Default color


# Plotting the DataFrame with a color scale
plt.figure(figsize=(10, 6))
plt.imshow(clustersBestK.values, cmap='viridis', aspect='auto')

# Set ticks and labels
plt.xticks(ticks=np.arange(len(clustersBestK.columns)), labels=clustersBestK.columns)
plt.yticks(ticks=np.arange(len(clustersBestK)), labels=np.arange(len(clustersBestK)))

# Add colorbar
plt.colorbar(label='Value')

plt.title('DataFrame with Color Scale')
plt.xlabel('Cluster number')
plt.ylabel('Attributes (columns of the DataFrame)')
plt.show()

## Lastly we will print which attributes have been most valuable in clustering for each cluster center:
print("MOST IMPORTANT COLUMNS FROM EACH CLUSTER")
for i in range(0, bestK):
    print("Cluster number " + str(i) + ":")
    print(clustersBestK.iloc[i].sort_values(ascending=False)[:5])
    print("============================")

## We can see that different clusters pick different attributes: some pick sports, some pick software, some pick 3d
## games, 2d games, rpgs... so, in fact, it is doing the clustering that we expected from the beginning,
## at a very reduced scale.

## Unfortunately, the games that I picked by hand to "test" the model where mostly fitted into one of the two
## biggest clusters, which contain most of the games from the list. The proposed method isn't very effective when
## plotting those games.

## By looking at the Elbow Method we can see that the inertia is generally reduced each time that we increase the "k"
## value, and it looks like it would still not reach an "asymptote". We can tell that the "k" value should be
## even higher, but it's beyond the capabilities of my computer.

## Overall, it can be seen that this model is not that simple. 100 clusters hardly gives a good result, as there are
## two enormous clusters and the rest consists of only specific games. With a higher number of clusters,
## this idea could work better, but my computer doesn't have the power to handle it. Furthermore, the complexity of
## similarity between games doesn't only reside in the categories of the games themselves, but we should also take
## into account what other players have played that is similar in order to reinforce the links between games (maybe
## two games are located in different categories but their players are the same!). Overall, the graphs show that the
## clustering process isn't ideal and that the number of clusters that we would need would have to be a lot higher.
