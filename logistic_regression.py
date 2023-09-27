# LOJİSTİK REGRESYON


""""
- Bağımlı değişken ikili olduğunda gerçekleştirilecek uygun regresyon analizidir.
- Tüm regresyon analizleri gibi lojistik regresyon da tahmine dayalı bir analizdir.
Verileri tanımlamak ve bir bağımlı bir bağımlı ikili değişken ile bir veya daha fazla
nominal, sıralı, aralıklı veya oran düzeyinde bağımsız değişken arasındaki ilişkiyi
açıklamak için kullanılır.
- İkili değer alması 0 ve 1'lerden oluştuğu şeklinde söylenebilir. Örneğin,
canlı olup olmaması, dersten geçip geçmeme durumu gibi.
- Lojistik regresyon, adını matematiksel olarak sigmoid fonksiyonun kullanılmasından
alır. Sigmoid fonksiyonu, girdi değerlerin, 0 ile 1 arasında sıkıştıran bir
fonksiyondur ve bu özelliği ile olasılık değerlerini ifade etmek için uygundur.

Sigmoid fonksiyonu: yi = 1 / 1 + e¹-(z) şeklindedir.

Örneğin;
Verilen bias ve weight'lere göre aşağıdaki gözlem birimi için 1 sınıfına
ait olma olasılığını hesaplayınız.

b = 5, w1 = 4, w2 = -4, w3 = 3; x1 = 2, x2 = 3, x3 = 0
z = b + w1x1 + w2x2 + w3x3
z = 5 + 4*2 + -4*3 + 3*0 = 5 + 8 - 12 + 0
z = 1
1 / 1 + e(-¹) = 0.731


# SINIFLANDIRMA PROBLEMLEMLERİNDE BAŞARI DEĞERLENDİRME KRİTERLERİ

ACCURACY(doğruluk): Accuracy, bir makine öğrenimi modelinin performansını
değerlendirmek için kullanılan bir ölçüttür. Doğru tahmin edilen örneklerin
sayısının toplam örnek sayısına oranının yüzdesidir. Örneğin, bir model 100
tahminden 70'ini doğru tutturduysa modelin doğruluk oranı %70'dir.
PRECISION: Pozitif olarak tahmin edilen örneklerin gerçekte pozitif olup
olmadığını ölçer. Örneğin, bir spam filtresi için spam olarak tahmin edilen
e-postaların gerçekte spam olup olmadığını ölçer.
RECALL: Gerçek pozitif vakaların doğru bir şekilde tanımlanan yüzdesidir.
Örneğin, bir kanser tespiti modeli için, kanserli olarak tahmin edilen vakaların
gerçekte kanserli olup olmadığını ölçer.
- İkisini içeren bir örnek olarak sahtekarlık tespiti verilebilir.
Eğer sahtekar olmayan birisine sahtekar denir ve bunun oranı ölçülürse
precision kullanılır. Sahtekar olup gözden kaçırılan oran ise recall kullanılır.

CONFUSION MATRIX(KARMAŞIKLIK MATRİSİ):
- Bir sınıflandırma modelinin performansını değerlendirmek için kullanılan
bir tablodur. Matris, modelin gerçek pozitif(TP), gerçek negatif(TN),
yanlış pozitif(FP), yanlış negatif(FN) değerlerini gösterir.
ACCURACY = (TP + TN) / (TP + TN + FP + FN)
PRECISION = TP / (TP + FP)
RECALL = TP / (TP + FN)
F1 SCORE = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)

ÖRNEK: Bir bankada 1000 kredi kartı işlemi yapılmıştır. 990 normal işlem
ve 10 sahtekar işlem yapılmıştır. 990 normal işlemden 900'ü doğru tahmin
edilirken 10 sahtekar işlemden 5'i doğru tahmin edilmiştir. Buna göre
başarı metriklerini hesaplayınız.

Accuracy = (900 + 5) / 1000 = 0.905
Precision = 5 / (5 + 90) = 0.05
Recall = 5 / (5 + 5) = 0.50
F1 - SCORE = 2 * (0.05 * 0.50) / (0.05 + 0.55) = 0.09

CLASSIFICATION THRESHOLD (SINIFLANDIRMA EŞİĞİ)
- Lojistik regresyonda sınıflandırma eşiği, modelin bir örneği pozitif veya
negatif olarak sınıflandırmak için kullandığı eşiktir.
- Örneğin, sınıflandırma eşiği 0.60 ise bu orandan büyük olanlar pozitif
yani 1, bu orandan az olanlar ise negatif yani 0 şeklinde sınıflandırılır.
Lojistik regresyonda ön tanımlı eşik değeri 0.50'dir.

LOG LOSS
- Lojistik regresyonda, modelin tahmin ettiği olasılık değerleri ile gerçek değerler
arasındaki farkı ölçen bir ölçüttür. İkili sınıflandırma problemleri için
kullanılır. Bir örneğin belirli bir sınıfa ait oloma olasılığını tahmin etmek
için kullanılır. Bu olasılık, 0 ile 1 arasında bir değerdir.
- Log loss formülü,
-1 / N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
- N, örnek sayısını temsil eder.
- y_true, gerçek değerleri temsil eder.
- y_pred, modelin tahmin ettiği olasılık değerlerini temsil eder.
- Log loss değeri ne kadar düşükse model o kadar iyi bir performans gösterir.

"""

"""
DIABETES PREDICTION WITH LOGISTIC REGRESSION 

Problem: Özellikleri belirtildiğinde kişilerin diyabet hastası olup
olmadıklarını tahmin eden bir makine öğrenmesi geliştirebilir misiniz?

Veri seti: "diabets.csv" dosyası ABD'deki Ulusal Diyabet-Sindirim-Böbrek 
Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
Arizona Eyaleti'nin en büyk 5. şehri olan Phoneix şehrinde yaşayan 21 yaş  
ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması
için kullanılan verilerdir. 768 gözlem ve 8 sayısal bağımsız değişken üzerinde
oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet 
test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir. 

DEĞİŞKENLER 
Pregnancies: Hamilelik sayısı
Glucose: Glikoz
BloodPressure: Kan basıncı
SkinThickness: Cilt kalınlığı
Insulin: İnsülin
BMI: Beden kitle indeksi
DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma 
olasılığını hesaplayan bir fonksiyon
Age: Yaş (yıl)
Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) değilse (0)


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)
pd.reset_option('display.width', 500)

def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# EXPLORATORY DATA ANALYSIS (KEŞİFÇİ VERİ ANALİZİ)

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape

# Target (hedef değişken) analizi
df["Outcome"].value_counts()

# görselleştirme
sns.countplot(x = "Outcome", data = df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

# Feature (bağımsız değişkenlerin) analizi

df.describe().T

df["Glucose"].hist(bins = 20)
plt.xlabel("Glucose")
plt.show()

# görselleştirme

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins = 20)
    plt.xlabel(numerical_col)
    plt.show(block = True)

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col] # sadece bağımsız değişkenler

for col in cols:
    plot_numerical_col(df, col)

# Target vs Features

df.groupby("Outcome").agg({"Pregnancies": "mean"})

# Bağımlı değişkenin bağımsız değişkenlerle ilişkisi

df.groupby("Outcome").agg({"Pregnancies":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end= '\n\n\n')

for col in cols:
    target_summary_with_num(df, "Outcome", col)

df.isnull().sum()

df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_threshold(df, "Insulin")

# scaler

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

# MODEL & PREDICTION

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X,y)

log_model.intercept_ # sabit
log_model.coef_ # bağımsız değişkenlerin ağırlıkları

y_pred = log_model.predict(X)

y_pred[0: 10]
y[0:10]

# Model Evulation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score {0}'.format(acc), size = 10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Prediction: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y, y_prob) # 0.83939
