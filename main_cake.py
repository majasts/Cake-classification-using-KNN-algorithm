import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import math

data = pd.read_csv('cakes.csv')
pd.set_option('display.max_columns',13)
pd.set_option('display.width', None)

data['eggs'] = data['eggs']*63

#-----1.stavka-----

print(data.head(5)) #ispis prvih 5 vrsta

#-----2.stavka----

print(data.info())
print(data.describe())
print(data.describe(include=[object]))

# provera prisutnosti nan vrednosti
total = data.isnull().sum().sort_values(ascending=False)
perc1 = data.isnull().sum() / data.isnull().count() * 100
perc2 = (round(perc1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, perc2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))

# -----3. STAVKA------

# Nema Nan vrednosti

#----4.stavka----

plt.figure()
num_data = data.select_dtypes(include=np.number)
del num_data['eggs']

mat = num_data.corr()
plt.figure(figsize=(10,10))
sb.heatmap(mat,
             annot=True,
             fmt='.2f',
             annot_kws={'fontsize': 10}
           )
plt.xticks(fontsize=9,rotation=45)
plt.yticks(fontsize=9, rotation=30)
plt.show()

# ----- 5.STAVKA -----

for i in range(0, len(num_data.columns)-1):
    plt.figure()
    plt.scatter(num_data.iloc[:,i], data.iloc[:,-1])
    plt.xlabel(num_data.columns[i])
    plt.ylabel(data.columns[-1])
    plt.show()

# ----- 6. STAVKA -----

cat_data = data[['eggs', 'type']]
sb.catplot(data = cat_data, y='eggs', x='type', kind='bar')
plt.show()

# ----- 7. STAVKA ------

X = data[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]

#kodiranje vrednosti
enc = OrdinalEncoder()
Y = enc.fit_transform(data[['type']])

# skaliranje podataka
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))

# podela na skup u kom odredjujemo izlaznu vr. i skup nad kojim gledamo susedstvo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     test_size=0.2,
                                                     shuffle=True,
                                                     random_state=1)

# izbor vrednosti za k
K = round(math.sqrt(len(X_train)))
if len(X_train) % 2 == 0:
    K += 1

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=K)
knn_model.fit(X_train, Y_train)
Y_pred = knn_model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = math.sqrt(mse)

print('KNN: Rmse: ', rmse)
print('KNN: score: ', accuracy_score(Y_test, Y_pred), end='\n')
Y_pred_string = np.where(Y_pred == 0, 'cupcake', 'muffin')
Y_test_string = np.where(Y_test == 0, 'cupcake', 'muffin')
print('KNN: Predicted output:', Y_pred_string)
print('KNN: Real output:', np.transpose(Y_test_string))

#moja KNN fja

from scipy.stats import mode

def eucledian(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist

def predict(x_train, y_train, x_test, k):
    y_pred = []

    for i in range(len(x_test)):
        curr_item = np.array(x_test.iloc[i,:])
        point_dist = []
        for j in range(len(x_train)):
            curr_neigh = np.array(x_train.iloc[j, :])
            distances = eucledian(curr_neigh, curr_item)
            point_dist.append(distances)
        point_dist = np.array(point_dist)
        dist = np.argsort(point_dist)[:k]
        labels = y_train[dist]
        label_pred = mode(labels, keepdims=True)
        label_pred = label_pred.mode[0]
        y_pred.append(label_pred)

    return y_pred

from sklearn.model_selection import KFold

kfold = KFold(5, shuffle=True, random_state=1)
data_final=np.concatenate([X, Y], axis=1)
k=1
for train, test in kfold.split(data_final):

    X_train = X.iloc[train, :]
    X_test = X.iloc[test, :]
    Y_test = Y[test, :]
    Y_train = Y[train,:]
    Y_pred = predict(X_train,Y_train,X_test , K)
    test_mse = mean_squared_error(Y_test, Y_pred)
    test_rmse = math.sqrt(test_mse)
    print('My KNN: Fold: %2d, Accuracy: %.3f' % (k , accuracy_score(Y_test, Y_pred)))
    scores=[]
    scores.append( accuracy_score(Y_test, Y_pred))
    print('My KNN: Fold: %2d, rmse: %.3f' % (k + 1, test_rmse))
    k += 1

print('\n\nCross-Validation accuracy: %.3f ' % (np.mean(scores)),'\n \n')

Y_pred = np.array(Y_pred)
Y_pred_string = np.where(Y_pred == 0, 'cupcake', 'muffin')
Y_test_string = np.where(Y_test == 0, 'cupcake', 'muffin')
print('My KNN: Predicted output:', np.transpose(Y_pred_string))
print('My KNN: Real output:', np.transpose(Y_test_string))
