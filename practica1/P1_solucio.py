# -*- coding: utf-8 -*-

# For solving this practic exercise you will need to use:
#   python programming language 
#   pandas library (pandas.pydata.org) -> Tutorial ... http://pandas.pydata.org/docs/getting_started/index.html
#   sikit-learn library
#
# Is recomended to use anaconda (www.anaconda.com) and use built in spyder IDE.
# As with a single instalation all the required libraries will be instalated.
# All computers in the laboratory has this optin and others (pycharm...) already installed


#1) Load the starwars dataset as a pandas dataframe. (pandas.pydata.org) (1 point)

#resposta esperada.
import pandas as pd
data=pd.read_csv("star_wars_character_dataset.csv")

#d'on surt la resposta.
"""
Aqui s'ha d'importar la llibreria que utilitzarem i carregar el datasset amb la funcio read_csv
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

Segur que hi ha altres opcions igual de correctes pero aquesta seria la mes directe i facil.
"""


#2) Inspect de dataser. How many intstances and variales there are in the original dataset? (1 point)

#resposta
#87 instances and 14 variables

#d'on surt la resposta.
"""
Aqui podiem des de mirar-ho a la nostre IDE fins a utilitzar funcions de pandas o python per aconseguir-ho.
exemples de codis que ens donen l'informacio que necesitem.

data.shape

o be

data.head()
"""

#3) Let's imagine we want to create a model that predicts the homeworld of a character.
# Do you think that all the variables are relevant? Wich ones are important and wich ones are not in our case.
#If some of them are not relevant delete them. Give the reason why you delete the variables. (1 point)


# exemple de resposta Com que soc un friki de starwars, he vist totes les pelicules i series puc assegurar que
# aquestes variables no son relevants. Les elimino.
data=data.drop(['name','films','vehicles','starships'],axis=1)

# eliminem els nans no es demana a aquest apartat pero per mes endavant ho necesitem perque son registres amb masses
# buits i no aporten res....
data=data.dropna()



#d'on surt la resposta.
"""
Aquí cal fer la selecció de variables. No hi ha un criteri unic per fer-ho.... A continuacio els mes comuns que ens trobem quan treballem amb dades.
Sempre tindrem a l'avast diverses opcions, depenent del cas haurem d'escollir la que més ens convé o combinar opcions:
    1) Coneixement expert - Tenim informació o coneixements sobre el tema i saben que certes variables son o no rellevants per tant ho apliquem directament. 
                            Si som uns experts en Star Wars i ja sabem quines son les variables importants ho apliquem, perque buscar un coneixement que ja tenim???
                            Aqui, podem graficar les dades per entendre com son. O be fer plots d'una variable contra l'altre per trobar correlacions, histogrames per veure distribucions...
                            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

    2) Metodes estadistics - Qualsevol analisi estadistic de les dades ens pot servir per saber com son les nostres dades i si alguna variable pot no ser util.
                            La metodologia mes simple d'aplicar si tenim les dades en un dataframe de pandas es la matriu de correlacions.
                            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
                            
    3) Prova/ error - Variables que mes endavant detectem que ens fan anar malament degut a les seves caracteristiques (Masses nans, poc correlacionades,...)    
"""


#4) In datamining, statistics, and mathematics we can talk about predictor variables (X) and predicted/objective variables (y).
#Split the dataset in X and y datasets. (1 point)

#exemple de resposta
y= data['homeworld']
X= data.drop('homeworld',axis=1)#the class must be deleted!! from X


#d'on surt la resposta.
"""
La sikitlearn (llibreria que utilitzarem en aquesta assignatura per fer els models) necessita les dades partides en X variables predictores i la y les predites. Cal simplement partir el dataset en 2. 
    Per fer ho tenim diverses opcions:
    https://pandas.pydata.org/docs/user_guide/indexing.html

Aqui tambe cal recordar que a la matriu X no hi ha d'haver la variable que volem predir.
    Podem utilitzar  per eliminar si ens cal:
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    
"""


#5)In datmining the model is trained with one datassed and tested with new unseen data. Prepare the train and test datasets.
# Which test % leaves sikitlear by default? Change it to use only 30%? (1 points)

#resposta d'exemple
# By default, 25%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3 )


#d'on surt la resposta.
""" 
Tot i que es pot fer d'altres maneres. La mes facil es utilitzar les funcions que ens proporciona scikit-learn.

Calia anar a la api de sikit per saber quina funcio necesitavem. Quin parametre calia canviar i quin era el valor per defecte.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""



#6) Build and train the DummyClassifier with strategy="most_frequent" and a GaussianNB models using the previously created Train datasets. (3 points)

#resposta

#primer codifiquem els atributs categorics a numerics
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(X) #si el transformer no coneix totes les categories no pot transformar

#Creem els models que ens demanen
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(enc.transform(x_train), y_train)
y_pred_Dumy = dummy_clf.predict(enc.transform(x_test))#aui faig ja la prediccio del test. No es demana pero es necesita al seguent apartat. Aqui queda mes net.

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(enc.transform(x_train), y_train)
y_pred_NB = gnb.predict(enc.transform(x_test))#aui faig ja la prediccio del test. No es demana pero es necesita al seguent apartat. Aqui queda mes net.


#d'on surt la resposta.
"""
Aqui cal crear els 2 models que es demanen. Cal crear el Dumyclasifier i el GausianNB.

El copy-Paste directe dels exemples de la sikitlern no funcionara.
O be hem de llegir la documentacio per saber que vol l'algorisme o be hem de ser capaços de interpretar els errors que ens surten i solucionar-los.

Aqui ens hem d'adonar (be interpretant l'error o be mirant les api) que aquests algorismes nomes poden treballar amb atributs tipus numeric i no suporten observacions amb buits.

La solucio aqui tampoc sera unica, i no ni ha una de mes valida que les altres. Com sempre la solucio bona dependra del cas concret. Per aquesta practica qualsevol solucio seria correcte.
Solucions mes comunes quan ens trobem amb aquest problema:
    1)Eliminar el que no podem tractar. Pot ser una bona solucio quan tenim moltes dades o variables redundants.
            Punt fort. No introduim soroll ni dades falses.
            Punt feble. Perdem informacio.
            Com ho fem:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
            
    2)Transformar les dades per poder les tractar.
            Punt fort. No perdem infromacio.
            Punt feble. Estem "falsejant" les dades. Podem arribar a casos on el model ens apren nomes lo que hem afegit en imputacions com a comportament bo. O altres situacions estranyes.
        Per tractar els nans per exemple tenim:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
         
        Per passar strings a numeric per exemple tenim:
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
            
            A la seccio de preprocessat de la scikit tenim moltes mes opcions....
            
    3)Buscar un algorisme que ens permeti tractar les dades que tenim. En aquest cas estavem limitats ja que haviem de fer un NV i el dumy classifier. 
        Pero tot i estar limitats a la sikitlearn tenim diverses implementacions de Naive bayes. Si es justifica correctament el canvi d'algorisme no es lo que espero que feu pero podria considerar-ho correcte.
        Per atributs categorics: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
        Podem utilitzar-lo i passar els numerics a string... 
"""


#7) Compare the scores and conusion matrixes of the 2 models. Results are bad with both techniques.
# Naibe bayes is a probabilistic model predicts the most probable ansher acording to the training set.
# DummyClassifier with mostcommon strategy predicts allways the most common class seen in the training dataset.
# Visualize both confusion_matrix. And accuracy_score.
# Explain what's happening here. (3 points)

#Possible Resposta:
#Com es d'esperar DummyClassifier prediu sempre la classe majoritaria en el test. NB canvia en funcio de la probabilitat condicionada.
#Els 2 models fallen i van malament. A priori podem pensar que es perque son models massa "Tontos" pero no es aixi.
#Aqui el que passa son varies coses molt comunes a el mon real.
# 1 que els atributs que tenim no tenen quasi cap correlacio amb la classe per tant es quasi impossible crear un bon model.
# 2 les classes estan desbalançejades (una de les classes te moltissimes observacions, l'altre poques) la majoritaria es veura sempre afavorida si no fem res per evitar-ho
# 3 tenim molt poques dades. I les poquies que tenim tenen molts Buits...
from sklearn.metrics import confusion_matrix
dum=confusion_matrix(y_test,y_pred_Dumy)
nb=confusion_matrix(y_test,y_pred_NB)
print(dum)
print(nb)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_Dumy))
print(accuracy_score(y_test,y_pred_NB))

#d'on surt la resposta.
"""
Els passos anteriors poden afectar el que ens trobem en aquest punt sobretot si hem eliminat moltes variables i/o moltes observacions podem estar sense dades. I fer models de dades sense dades es impossible....
(no tothom ha de tenir les mateixes conclusions, ja que el resultat canvia segons els passos previs)

Aqui el que hem de veure es que els 2 algorismes ens estan fent el mateix o quasi el mateix. I que cap dels 2 comja es diu a l'enunciat va massa be.

Si mirem mes al detall les matrius de confusio veurem que El dumy ens prediu sempre la classe majoritaria (com ha de ser) i el Naibe bayes fa tambe quasi lo mateix prediu la classe majoritaria i els pocs cops que no la prediu s'equivoca.
Perque ens esta passant aixo. 
1 Els atributs que tenim estan poc correlacionats amb la classe.
2 Les classes estan desbalancejades (hi ha una classe amb moltes observacions i altres amb molt poques)
3 Tenim poques dades sobretot d'algunes classes. I variables amb molts de nans

Es un problema molt dificil de resoldre i amb el datasset variables proporcionades podriem dir que es impossible d'aconseguir un model que faci be la classificacio.
"""


#8) Change the datasset and do the same steps for the bank dataset. Here both techniques get good results can you explain why? (6 points)

#Resposta exemple

"""
Aqui s'ha de veure que al carregar el dataset no es la configuracio per defece aom avans ja que el CSV te ; en comptes de ,
Tambe cal especificar-ho al parametre sep=';' tambe tenim que els nans estan marcats com a "unknown" 
si volem eliminar-los haurem de especificar na_values='unknown' perque ens els carregui com a nan.

La resta es purament copy paste del que ja hem fet.
"""
data=pd.read_csv("bank-full.csv", sep=';', na_values='unknown')
data=data.dropna()

y= data['y'] # Aqui cal llegir bank-info.txt per saber que estem predint i perque!!!
X= data.drop('y',axis=1)#class deleted from X

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3 )

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(X) #si el transformer no coneix totes les categories no pot transformar

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(enc.transform(x_train), y_train)
y_pred_Dumy = dummy_clf.predict(enc.transform(x_test))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(enc.transform(x_train), y_train)
y_pred_NB = gnb.predict(enc.transform(x_test))

from sklearn.metrics import confusion_matrix
dum=confusion_matrix(y_test,y_pred_Dumy)
nb=confusion_matrix(y_test,y_pred_NB)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_Dumy))
print(accuracy_score(y_test,y_pred_NB))

"""
Interpretacio de resultats. Lo que realment es important i es valorariaper obtenir punts aqui.
En quan els resultats si mirem nomes l'acuracy veiem que aparentment els resultats son molt bons (trampa), si mirem la matriu de confusio veurem que per una de les 2 classes (la majoritaria) si que ho esta fent be, pero per l'altre no. Podriem calcular el TPR o FPR per quantificar mes exactament o amb una simple exploracio de la matriu ho podem observar.
Aquest dataset esta desbalancejat (una de les classes te moltes mes observacions que l'altre) per tant l'algorisme tendeix a afavorir la classe majoritaria. Si ho volem arreglar hauriem de valancejar el dataset.
Aixo ens passa amb aquests algorismes "simples" pero es un comportament comu de qualsevol altre algorisme mes avançat.
"""