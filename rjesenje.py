import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Učitavanje podataka iz CSV datoteke
data = pd.read_csv('train_data.csv')

# Stvaranje matrice značajki i vektora ciljne varijable
X = data[['KATEGORIJA', 'GODINA', 'TJEDAN']]
y = data['PRODAJA']

# Stvaranje preprocesorskih transformacija
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('onehot', onehot_transformer, ['KATEGORIJA']),
    ('num', numerical_transformer, ['GODINA', 'TJEDAN'])
])

# Stvaranje i treniranje modela
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LinearRegression())
])

model.fit(X, y)

# Izračunavanje predikcije prodaje za nove podatke
novi_podaci = pd.DataFrame({'KATEGORIJA': ['kategorija_1'], 'GODINA': [2023], 'TJEDAN': [12]})
predikcija = model.predict(novi_podaci)

import matplotlib.pyplot as plt

# Stvaranje grafa
plt.plot(data['TJEDAN'], data['PRODAJA'], 'o')
plt.plot(novi_podaci['TJEDAN'], predikcija, 'x')

# Dodavanje oznaka osi i naziva grafa
plt.xlabel('TJEDAN')
plt.ylabel('PRODAJA')
plt.title('Predikcija prodaje')

# Prikazivanje grafa
plt.show()

print(predikcija)


# Definiranje godina, tjedana i kategorija
godine = [2023]
tjedni = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
kategorije = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

# Stvaranje praznog DataFrame-a za spremanje predikcija
predikcije_df = pd.DataFrame(columns=['KATEGORIJA', 'GODINA', 'TJEDAN', 'Predikcija'])

# Izračunavanje predikcija za sve kombinacije godina, tjedana i kategorija
for godina in godine:
    for tjedan in tjedni:
        for kategorija in kategorije:
            novi_podaci = pd.DataFrame({'KATEGORIJA': [kategorija], 'GODINA': [godina], 'TJEDAN': [tjedan]})
            predikcija = model.predict(novi_podaci)[0]
            predikcije_df = predikcije_df.append({'KATEGORIJA': kategorija, 'GODINA': godina, 'TJEDAN': tjedan, 'Predikcija': predikcija}, ignore_index=True)

# Prikazivanje predikcija u obliku tablice
print(predikcije_df)

from pathlib import Path  
filepath = Path('out.csv')
predikcije_df.to_csv(filepath)  

