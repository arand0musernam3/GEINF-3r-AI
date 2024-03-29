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

# Load the dataset in "steam.csv". We want to perform a clustering using k-means, scikit-learn only implements
# Euclidean distance so only accepts numeric attributes, transform all the attributes to numeric. Preprocess the data
# in other to convert categorical and non-numeric attributes to numeric. (1 point)

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


data = read_model()

# Fill the NaN values in the dataset. Use A KNN algorithm. Was a good decision to fill these
# values using KNN or is better to give them a fix value or just delete them? What will
# happen in each case? (3 points)

## In order to fill the NaN values we need to use a KNN algorithm. The following code does that:

data['publisher'] = data['publisher'].replace(le_publisher.transform(["THIS_IS_A_NAN_VALUE_REPLACE_ME"])[0],
                                              np.nan)  # getting back the np.nan values
data['developer'] = data['developer'].replace(le_developer.transform(["THIS_IS_A_NAN_VALUE_REPLACE_ME"])[0], np.nan)
imputer = KNNImputer()
dataProcessed = imputer.fit_transform(data)
dataProcessed = pd.DataFrame(dataProcessed, columns=data.columns)
print(dataProcessed)


## The generated values for the NaN instances don't make much sense, as "similar" games needn't have similar developers.
## An example would be a shooter like Battlefield or Call of Duty, both similar in terms of gameplay but with
## different publishers and developers. It would have made more sense to generate a new value "Unknown", which could
## encapsulate the lack of knowledge without compromising the data with false associations.

## It must be noted that the "Unknown" value also affects the games whose developers and publishers are NaN,
## as they would have a "fake" similarity in this matter. Another option could have been to manually determine its
## developer or, as a last resort, dropping the instances (which are not many) could still leave a valuable dataset to
## work with.

## *Note:* This should have probably been done AFTER scaling the data, not BEFORE.
## The results would have been more concise as distances between variables would be scaled.

# Clustering needs data to be scaled. If not variables with higher variability will be favoured.
# Scale the data. (1 point)


def scaleColumns(df):
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(df))
    scaled_features.columns = df.columns
    return scaled_features


dataProcessed = scaleColumns(dataProcessed)
print(dataProcessed)

# Fit a k-means clustering with 2 clusters and perform the prediction. Retrieve the cluster
# centers (cluster_centers_ attribute on your fitted k-means). Try to understand what kind
# of information each cluster has. (2 points)


# TODO: mirar quin preprocessat més es pot fer, fer gràfics de resultats, etc. donar una explicació raonable
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a


# =====================================================================================================
model = KMeans(n_clusters=2, init='k-means++')
model.fit(dataProcessed)
label = model.predict(dataProcessed)
u_labels = np.unique(label)
## The model is now fitted. It is time to determine which information each cluster holds.

## A first approach could be visualizing the data given two axis of the dataset to try and see which differentiates the
## values in a clearer way.

# for i in dataProcessed.columns:
#     if i == 'appid':
#         continue
#     plt.scatter(dataProcessed['appid'], dataProcessed[i], c=label, cmap='plasma')
#     plt.xlabel('appid')
#     plt.ylabel(i)
#     plt.show()

## After plotting all attributes, we can see that the most relevant ones are 'developer' and 'publisher'. Let's create
## another 2d plot where the values are correlated.


plt.scatter(dataProcessed['developer'], dataProcessed['publisher'], c=label, cmap='plasma')
plt.xlabel('developer')
plt.ylabel('publisher')
plt.show()

## https://imgur.com/a/ngsedpm

## It can be seen that there is a clear separation between the two attributes. The diagonal line implies that lots
## of the games were published and developed by the same people.

correlation = dataProcessed.corr()

print(correlation)  # https://imgur.com/a/GSsCjGw

## We can see that, in fact, these attributes are quite similar and could even be discarded to some extent. The fact
## that this model is not supervised means that we probably shouldn't, and since it is not specified by the questions
## it will not be done.


## Taking a look at the 3d visualization of the

import numpy as np
import matplotlib.pyplot as plt

#i, j = 1, 1  # missing 'appid' on purpose at it has no real meaning
#while i < len(dataProcessed.columns):
#    j = i + 1
#    while j < len(dataProcessed.columns):
#        plt.xlabel(dataProcessed.columns[i])
#        plt.ylabel(dataProcessed.columns[j])
#        plt.scatter(dataProcessed[dataProcessed.columns[i]], dataProcessed[dataProcessed.columns[j]],
#                    c=label, cmap='plasma')
#        plt.savefig("./output/" + dataProcessed.columns[i] + "-" + dataProcessed.columns[j] + ".svg")
#        plt.clf()
#        j = j + 1
#    i = i + 1

# =====================================================================================================


# Set yourself a clustering goal. Do you see any not useful variable? delete it. Try other
# cluster numbers, preprocessing, variable selections, add weights add information in
# other files... (do what you want to achieve a goal). Try to explain what are you expecting
# to achieve, what are you doing and what you achieve. (3 points)

## The main idea is to cluster the different video games by two main terms: their respective areas and the price per
## fun. The ideal would be to have clustered the games which are both more fun from a specific category, because one
## of the main struggles of gamers is finding a good, similar game to another one you played and enjoyed. This is what
## this clustering aims to do.

## Setting up the data from all the tables.

steamData = read_model()
steamTagData = pd.read_csv('steamspy_tag_data.csv')

## We need to combine both into one big dataset. Before, we'll get rid of unnecessary variables which are not correlated
## at all.



correlation = steamData.corr()
print(correlation)  # https://i.imgur.com/x85MUWZ.png

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
print(mergedData)

mergedData = mergedData.drop(columns=["name", "release_date", "developer", "publisher", "required_age", "categories",
                                      "genres", "achievements", "median_playtime", "owners"])

## now we need to scale the data for the distances to work properly

mergedData = mergedData.dropna()  # we might lose some data, but we've got more than enough to get a decent study

scaleColumns(mergedData)

## PROVAREM AMB DBSCAN PQ FA PINTA XUPI GUAI

from sklearn.cluster import HDBSCAN
k = 8
model5 = KMeans(n_clusters=k)  # HDBSCAN
labels5 = model5.fit_predict(mergedData)
print(labels5)

## https://developers.google.com/machine-learning/clustering/interpret

## Cluster cardinality
unique_labels5 = np.unique(labels5)
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
plt.show()

plt.clf()

## Cluster magnitude
### Sum of distances from all examples to the centroid of the cluster
plt.bar(unique_labels5, sum_labels5)
plt.xlabel("Cluster")
plt.ylabel("Sum of intracluster distances")
plt.title("Cluster magnitude")
plt.show()

## Magnitude vs cardinality

## Performance of similar measures
## TODO Buscar exemples de jocs semblants que jo conegui i veure si estan pròxims o no.
##  Valorar si val la pena fer Feature Selection primer!

## Optimum number of clusters
## TODO Amb HDBSCAN no n'hi ha, però amb Kmeans si!!

## KNOWN APPIDS (by own experience :) )
known_appids = {
    'shooter':[
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
    'strategy':[
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
    'car/truck simulators/games':[
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
    'singleplayer':[
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

print (known_appids)
