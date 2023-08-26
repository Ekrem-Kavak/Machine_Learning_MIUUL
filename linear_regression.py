# LINEAR REGRESSION (DOĞRUSAL REGRESYON)

"""
- Doğrusal regresyon, iki değişken arasındaki ilişkiyi tahmin etmek için kullanılan
bir istatistiksel analiz türüdür.
- Bağımsız değişken ile bağımlı değişken arasında doğrusal bir ilişki olduğunu
varsayar ve ilişkiyi tanımlayan en uygun doğruyu bulmayı amaçlar. Doğru, tahmin
edilen değerler ile gerçek değerler arasındaki farkların karelerinin toplamının en
aza indirilmesiyle belirlenir.

BASİT DOĞRUSAL REGRESYON (SIMPLE LINEAR REGRESSION)

- Basit doğrusal regresyonda bir bağımsız değişken ve bir bağımlı değişken vardır.
Model, değişkenler arasındaki ilişkiyi temsil eden en iyi uyum çizgisinin eğimini ve
kesişimini tahmin eder.
- Doğrusal regresyon, makine öğrenmesinde tahmine dayalı analiz için kullanılan sessiz
ve en basit istatistiksel regresyon yöntemidir. Doğrusal regresyon, bağımsız (tahmin edici)
değişken yani X ekseni ile bağımlı (çıktı) değişken yani Y ekseni arasındaki doğrusal
ilişkiyi gösterir ve doğrusal regresyon olarak adlandırılır. Tek bir giriş değişkeni X
(bağımsız değişken) varsa, bu tür doğrusal regresyona "basit doğrusal regresyon" adı
verilir.
- yi = b + wxi şeklinde gösterilir. Farklı notasyonları da vardır ancak mantık
katsayı ve ağırlık değerlerini hesaplamaktır.

GRADIENT DESCENT (GRADYAN İNİŞİ)

- Gradient descent, optimum minimum çözüme ulaşmak için maliyet (amaç) fonkisyonunu
optimize eden optimizasyon algoritmalarından biridir. Optimum çözümü bulmak için
tüm veri noktalarının maliyet fonksiyonunu (MSE) azaltmamız gerekir. Bu optimal bir
çözüm elde edene kadar B0 ve B1 değerlerinin yinelemeli olarak güncellemesiyle yapılır.
- Bir regresyon modeli, katsayı değerlerini rastgele seçerek ve ardından minimum maliyet
fonksiyonuna ulaşmak için değerleri yinelemeli olarak güncelleyerek maliyet fonksiyonunu
azaltarak hattın katsayılarını güncellemek için gradyan iniş algoritmasını optimize eder.
- Gradyan iniş algoritmasında, attığınız adım sayısı öğrenme oranı olarak düşünülebilir
ve bu algoritmanın minimuma ne kadar hızlı yakınsayacağını belirler.


"""
# SIMPLE LINEAR REGRESSION

# SALES PREDICTION WITH LINEAR REGRESSION

# Satış tahminleri içeren bir model oluşturalım.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" %x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]] # bağımlı değişken

# Model

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x

# sabit  (b - bias)
reg_model.intercept_[0] # 7.03

# TV'nin katsayısı (w1)
reg_model.coef_[0][0] # 0.047

# Tahmin

# 150 birimlik TV harcaması olsa ne kadar satış beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150 # 14.16

# 500 birimlik TV harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 500 # 30.80

df.describe().T

# NOT: Tablomuzda bulunan maksimum değerlerin dışındaki değerleri de
# kurduğumuz doğrusal regresyon sayesinde bulabiliriz.

# Modelin Görselleştirilmesi
g = sns.regplot(x = X, y=y, scatter_kws = {"color" : "b", "s": 9},
                 ci = False, color = "r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom = 0)
plt.show()

# Tahmin Başarısı

# MSE (Mean squared error - ortalama kare hatası)

y_pred = reg_model.predict(X) # tahmin edilen değerler

# Gerçek değer ile tahmin edilen değerlerin ortalama kare hatası
mean_squared_error(y, y_pred) # 10.51
y.mean() # 14.02
y.std() # 5.22

# RMSE (Root mean squared error - karekök ortalama hata)

# Ortalama kare hatanın karekökü alınır.
np.sqrt(mean_squared_error(y, y_pred)) # 3.24

# MAE (Mean absolute error - mutlak ortalama hata)

mean_absolute_error(y, y_pred) # 2.54

# R-SQUARE (R-kare)

# Bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesini ifade eder.
reg_model.score(X,y) # 0.61
# Bu modeldeki bağımsız değişken olan TV değişkeni, bağımlı değişken olan satış değişkenini
# %61 oranında açıklıyor şeklinde yorumlanmalıdır.

"""
NOT:
Tahmin başarısı için kullanılan yöntemler kendi içinde değerlendirilmelidir.
Birbirlerine üstünlükleri yoktur. Modele göre tercihler yapılabilir.
"""


# MULTIPLE LINEAR REGRESSION (ÇOKLU LİNEAR REGRESYON)

X = df.drop("sales", axis = 1) # sales değişkeni hariç hepsi bağımsız değişkendir.
y = df[["sales"]] # bağımlı değişken

# Model kurma

# Modelimizi train ve test olark ikiye ayırdık. %80'i eğitmek (train), %20'si test etmek için
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

X_train.shape # (160, 3) # 3 bağımsız değişken vardır.
y_train.shape # (160, 1)
X_test.shape # (40, 3)
y_test.shape # (40, 1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_[0] # 2.90

# coefficients (w - weights)
reg_model.coef_[0][0] # 0.04

# TAHMİN

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# Model denklemi -> sales = 2.90 + TV *0.04 + radio * 0.17 + newspaper * 0.002

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri) # 6.20

# Tahmin Başarısını Değerlendirme

# Train RMSE (Root mean squared error - ortalama kerekök hata)
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 1.73

# Train R-SQUARED (R - KARE)
reg_model.score(X_train, y_train) # 0.89
# Bağımsız değişkenlerin bağımlı değişkeni karşılama oranı %89'dur.

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 1.41

# Test R-KARE
reg_model.score(X_test, y_test) # 0.89

# 10 KATLI CROSS-VALIDATION (ÇAPRAZ DOĞRULAMA) RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv = 10,
                                 scoring = "neg_mean_squared_error"))) # 1.69

# 9 parçası ile model kurup 1 ile test etmek için kullanılır. Küçük veri setlerinde
# daha doğru sonuca ulaşmamıza olanak verir. 10 yerine farklı bir sayıda kullanılır.

# 5 KATLI CROSS-VALIDATION RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv = 5,
                                 scoring = "neg_mean_squared_error"))) # 1.71


# SIMPLE LINEAR REGRESSION WITH GRADIENT DESCENT (GRADYAN İNİŞİ)

# Cost function - MSE (maliyet fonksiyonu)
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# Ağırlıkların güncellenmesi
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - learning_rate * 1 / m * b_deriv_sum
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train (eğitim) fonksiyonu

def train(Y, initial_b, initial_w, learning_rate, num_iters):

    print("Starting gradient descent at a = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter = {:d} b = {:.2f}  w={:.4f} mse= {:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, learning_rate, num_iters)

