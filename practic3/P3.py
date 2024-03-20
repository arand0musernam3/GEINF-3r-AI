# NAME: Aniol Juanola Vilalta (u1978893)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


# Load the dataset in "steam.csv". We want to perform a clustering using k-means, scikit-
# learn only implements Euclidean distance so only accepts numeric attributes, transform
# all the attributes to numeric. Preprocess the data in other to convert categorical and non-
# numeric attributes to numeric. (1 point)

def read_model():
    df = pd.read_csv("steam.csv")
    df['release_date'] = pd.to_datetime(df['release_date']).astype('int64')
    # appid, name, release_date, english, developer, publisher, required_age, categories, genres, achievements,
    # positive_ratings, negative_ratings, average_playtime, median_playtime, owners, price
    le_name = LabelEncoder()
    le_developer = LabelEncoder()
    le_publisher = LabelEncoder()
    le_categories = LabelEncoder()
    le_genres = LabelEncoder()
    le_owners = LabelEncoder()
    df['name'] = le_name.fit_transform(df['name'])
    df['developer'] = le_developer.fit_transform(df['developer'])
    df['publisher'] = le_publisher.fit_transform(df['publisher'])
    df['categories'] = le_categories.fit_transform(df['categories'])
    df['genres'] = le_genres.fit_transform(df['genres'])
    df['owners'] = le_owners.fit_transform(df['owners'])



read_model()
# Fill the NaN values in the dataset. Use A KNN algorithm. Was a good decision to fill these
# values using KNN or is better to give them a fix value or just delete them? What will
# happen in each case? (3 points)

## In order to fill the NaN values we need to use a KNN algorithm. The following code does that:
model = KNeighborsClassifier()

# Clustering needs data to be scaled. If not variables with higher variability will be favoured.
# Scale the data. (1 point)

# Fit a k-means clustering with 2 clusters and perform the prediction. Retrieve the cluster
# centers (cluster_centers_ attribute on your fitted k-means). Try to understand what kind
# of information each cluster has. (2 points)

# Set yourself a clustering goal. Do you see any not useful variable? delete it. Try other
# cluster numbers, preprocessing, variable selections, add weights add information in
# other files... (do what you want to achieve a goal). Try to explain what are you expecting
# to achieve, what are you doing and what you achieve. (3 points)
