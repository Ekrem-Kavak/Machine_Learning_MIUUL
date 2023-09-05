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

"""

