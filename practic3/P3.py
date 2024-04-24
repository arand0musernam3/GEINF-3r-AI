# NAME: Aniol Juanola Vilalta (u1978893)
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# =====================================================================================================

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

if not os.path.exists("output/"):
    os.mkdir("output/")

if not os.path.exists("output/ex4"):
    os.mkdir("output/ex4")

if not os.path.exists("output/ex5"):
    os.mkdir("output/ex5")

# =====================================================================================================

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

# =====================================================================================================

# Clustering needs data to be scaled. If not variables with higher variability will be favoured.
# Scale the data. (1 point)


def scaleColumns(df):
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(df))
    scaled_features.columns = df.columns
    return scaled_features


dataProcessed = scaleColumns(dataProcessed)
print(dataProcessed)

## The data is now scaled between 0 and 1, so that differences between the different columns are the same.

# =====================================================================================================

# Fit a k-means clustering with 2 clusters and perform the prediction. Retrieve the cluster
# centers (cluster_centers_ attribute on your fitted k-means). Try to understand what kind
# of information each cluster has. (2 points)

## The model is now fitted. It is time to determine which information each cluster holds.

model = KMeans(n_clusters=2, init='k-means++')
model.fit(dataProcessed)
label = model.predict(dataProcessed)
u_labels = np.unique(label)
cluster_centers = model.cluster_centers_

print("These are the cluster centers: ")
print(cluster_centers)

print("These are the differences between the values of each cluster: ")
clustersDiffs = {}
for i in range(0, len(cluster_centers[0])):
    clustersDiffs[dataProcessed.columns.values[i]] = (abs(cluster_centers[0][i] - cluster_centers[1][i]))

sortedClusterDiffs = sorted(clustersDiffs.items(), key=lambda x: x[1], reverse=True)
print(sortedClusterDiffs)

## These are the cluster centers:
## [[5.43986610e-01 4.99980074e-01 8.93933430e-01 9.84082044e-01
##   4.93653679e-01 4.81964953e-01 2.14336647e-02 8.73972296e-01
##   3.78954399e-01 6.98635563e-03 3.05540161e-04 3.03354638e-04
##   7.35943495e-04 7.17489103e-04 1.64055618e-01 1.53813050e-02]
##  [5.76653232e-01 4.99796939e-01 8.92560684e-01 9.76925141e-01
##   5.02259740e-01 4.96235076e-01 1.72763319e-02 2.25835357e-01
##   3.89378027e-01 1.22556884e-03 4.81894744e-04 6.17905987e-04
##   8.56822156e-04 8.35432709e-04 1.18921981e-01 1.30138774e-02]]
## These are the differences between the values of each cluster:
## [('categories', 0.6481369386506869), ('owners', 0.045133637731246845), ('appid', 0.032666621908302274),
##  ('publisher', 0.014270123526929412), ('genres', 0.010423627450994288), ('developer', 0.008606060640007573),
##  ('english', 0.007156902674477217), ('achievements', 0.005760786785031495), ('required_age', 0.0041573328227511),
##  ('price', 0.002367427538161909), ('release_date', 0.0013727460787621437), ('negative_ratings', 0.0003145513490738414),
##  ('name', 0.00018313489276822192), ('positive_ratings', 0.00017635458314126324),
##  ('average_playtime', 0.0001208786613492701), ('median_playtime', 0.00011794360557845849)]


## It can be seen that the main difference in the clustering has been in the 'categories' column. Let's plot some graphs
## to see this in a visual manner.
import numpy as np
import matplotlib.pyplot as plt


def printGraphs():
    w, j = 1, 1  # missing 'appid' on purpose at it has no real meaning
    while w < len(dataProcessed.columns):
        j = w + 1
        while j < len(dataProcessed.columns):
            plt.xlabel(dataProcessed.columns[w])
            plt.ylabel(dataProcessed.columns[j])
            plt.scatter(dataProcessed[dataProcessed.columns[w]], dataProcessed[dataProcessed.columns[j]],
                        c=label, cmap='plasma')
            plt.savefig("./output/ex4/" + dataProcessed.columns[w] + "-" + dataProcessed.columns[j] + ".pdf")
            plt.clf()
            j = j + 1
        w = w + 1


printGraphs()

## The graphs show that, as expected, the plots which have the 'categories' column clearly show the difference between
## the clustering groups. It can be seen that the frontier between the values is located at the 0.5 mark
## approximately, which means that games from categories that have been normalized to lower values will be clustered
## together and separated from games that have been clustered with higher values.

## When looking at the other attributes, no clear evidence of clustering can be seen. We cannot extract conclusions
## from the graphs themselves.

## Let's also take a look at the correlation between variables:
correlation = dataProcessed.corr()  # https://imgur.com/a/GSsCjGw

## It can be seen that the 'categories' attribute has no real correlation with any other variable, so it makes sense
## that when grouping with this column the difference with the other attributes is relatively low.

## Lastly, we can use a PCA algorithm to reduce the dimension of the attributes and take a look at a 2D representation.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
dataReduced = pca.fit_transform(dataProcessed)
plt.scatter(dataReduced[:, 0], dataReduced[:, 1], c=label, s=50, cmap='plasma')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='*', s=200, c='#000000')
plt.xlabel('Main attribute 1')
plt.ylabel('Main attribute 2')
plt.title('KMEANS (Reduced by PCA)')
plt.show()

## Let's check which two attributes were used by the PCA:

attributes = list(dataProcessed.columns)

atr1 = attributes[pca.components_[0].argmax()]
atr2 = attributes[pca.components_[1].argmax()]

print("The main attributes from the PCA reduction are: ", atr1, " ", atr2)
## The main attributes from the PCA reduction are:  publisher   categories

## In this case, publisher also had a significant weight in the PCA reduction, even though the separation between
## the two centroids is minimal. We can assume that the publisher also had an important role in the clustering process.

# =====================================================================================================


# Set yourself a clustering goal. Do you see any not useful variable? delete it. Try other
# cluster numbers, preprocessing, variable selections, add weights add information in
# other files... (do what you want to achieve a goal). Try to explain what are you expecting
# to achieve, what are you doing and what you achieve. (3 points)

## The main idea is to cluster the different video games by their respective areas and categories into "k" big
## groups. The ideal would be to have clustered the games which are both more fun from a specific category,
## because one # of the main struggles of gamers is finding a good, similar game to another one you played and
## enjoyed. This is what # this clustering aims to do.

## Setting up the data from all the tables.
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
K = range(2, 100)  # the number of clusters could be increased even further
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
