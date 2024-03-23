# NAME: Aniol Juanola Vilalta (u1978893)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401

# Load the dataset in "steam.csv". We want to perform a clustering using k-means, scikit-
# learn only implements Euclidean distance so only accepts numeric attributes, transform
# all the attributes to numeric. Preprocess the data in other to convert categorical and non-
# numeric attributes to numeric. (1 point)

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
    df['developer'] = df['developer'].fillna("THIS_IS_A_NAN_VALUE_REPLACE_ME")  # Placeholder for future reference when dealing with NaNs
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

data['publisher'] = data['publisher'].replace(le_publisher.transform(["THIS_IS_A_NAN_VALUE_REPLACE_ME"])[0], np.nan)  # getting back the np.nan values
data['developer'] = data['developer'].replace(le_developer.transform(["THIS_IS_A_NAN_VALUE_REPLACE_ME"])[0], np.nan)
imputer = KNNImputer()
dataProcessed = imputer.fit_transform(data)
dataProcessed = pd.DataFrame(dataProcessed, columns=data.columns)
print(dataProcessed)

## TODO WRITE EXPLICACIÓ FINAL DE QUÈ EN PENSO: PROBABLEMENT HAURIA ESTAT MILLOR AFEGIR UN VALOR TIPU "UNKNOWN" I ALE :)

# Clustering needs data to be scaled. If not variables with higher variability will be favoured.
# Scale the data. (1 point)
### Buscar com normalitzar les dades.



def scaleColumns(df, cols_to_scale):
    scaler = MinMaxScaler()
    for col in cols_to_scale:
        df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df


scaleColumns(dataProcessed, dataProcessed.columns)
print(dataProcessed)

# Fit a k-means clustering with 2 clusters and perform the prediction. Retrieve the cluster
# centers (cluster_centers_ attribute on your fitted k-means). Try to understand what kind
# of information each cluster has. (2 points)
### No hi ha una forma predefinida, fer plots i osties i intentar endivinar-ho.

model = KMeans(n_clusters=2, init='k-means++')
model.fit(dataProcessed)
print(model.cluster_centers_)
#TODO: mirar quin preprocessat més es pot fer, fer gràfics de resultats, etc. donar una explicació raonable
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py


# Set yourself a clustering goal. Do you see any not useful variable? delete it. Try other
# cluster numbers, preprocessing, variable selections, add weights add information in
# other files... (do what you want to achieve a goal). Try to explain what are you expecting
# to achieve, what are you doing and what you achieve. (3 points)

###TODO: Mirar què es pot fer que sigui interessant, per exemple intentar agrupar per jocs similars, o per preu
### envers diversió si es pot, etc.
