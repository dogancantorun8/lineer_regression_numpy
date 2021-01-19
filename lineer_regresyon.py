import numpy as np

#conv_dict methoduyla her bir sütun üzerinde işlem yapıyorum

    #1.sütunun hepsine erişiyorum
    #2.sütunda NA olan değerler yerine "0" yazıyorum ve NA olmayanlar için metreye çevirme işlemi yapıyorum
    #3.sütunda NA olan değerler yerine "0" yazıyorum ve NA olmayanlar aynen alıyorum
    #4.sütunda önce tüm karakterleri küçültüyorum ve kadın olanlar için "1" erkek olanlar için return "0" ediyorum
conv_dict = {1: lambda s: s[1:-1], 2: lambda s: 0 if s == 'NA' else float(s) * 100, 3: lambda s: '0' if s == 'NA' else s, 4: lambda s: s.lower() == 'kadın'}

#loadtxt ile dosyayı okuyorum converters parametresinde sütun fonksiyonumu çağırıyorum
a = np.loadtxt('test.txt', delimiter=',', usecols=(1, 2, 3, 4), skiprows=1, encoding='utf-8', dtype='float32', converters=conv_dict)

#print(a)
#print()

#1.sütunda sıfır olmayan elemanların ortalamasını al ve 0 olan elemanların yerine ata
a[a[:, 1] == 0, 1] = np.round(np.mean(a[a[:, 1] != 0, 1]), 2)
#2.sütunda sıfır olmayan elemanların ortalamasını al ve 0 olan elemanların yerine ata
a[a[:, 2] == 0, 2] = np.round(np.mean(a[a[:, 2] != 0, 2]), 2)

#print(a)
#print()

#3 tane zeroslardan oluşan sütun tanımladım ve ana matrisimle concetenate fonksiyonuyla  birleştirdim
a = np.concatenate((a, np.zeros((a.shape[0], 3), dtype='float32')), axis=1)
#print(a)
#print()

#1.sütundaki ortalamadan büyük değerler için 4.sütuna bool 1 değerini ata
a[:, 4] = a[:, 1] >= np.mean(a[:, 1])
#2.sütundaki ortalamadan büyük değerler için 5.sütuna bool 1 değerini ata
a[:, 5] = a[:, 2] >= np.mean(a[:, 2])

#6.sütunda vücut kütle indexini hesapladım => VKİ = Ağırlık (Kg) / boyun metre cinsinden karesi
a[:, 6] = np.round(a[:, 2] / (a[:, 1] / 100) ** 2, 2)

#print(a)
#print()

import scipy.stats as stats
import matplotlib.pyplot as plt

#pearson korelasyon analizini kullanıyorum :boy ve kilo arasındaki korelasyonu(ilişkiyi) buluyorum
result = stats.pearsonr(a[:, 1], a[:, 2])
#print(result)
#print()

#y=mx+n ifadesindeki m ve n değerlerini linregress fonksiyonu ile bulup atama işlemi yapıyorum
result_all = stats.linregress(a[:, 1], a[:, 2])
#print(result_all)
#print()

#veri setimde boy ve kilonun saçılma fonksiyonunu çiziyorum
plt.scatter(a[:, 1], a[:, 2])
plt.show()

#165-200 arasında 100 tane eleman ataması yapıyorum
x_all = np.linspace(165, 200, 100)
#print(x_all)

#y=mx+n uyguluyorum x_all ile ürettiğim 100 tane değeri linregress fonk ile  veri setimden elde ettiğim m ve n değerlerini de yerine yazarak lineer regresyon uyguluyorum
y_all = result_all.slope * x_all + result_all.intercept
plt.plot(x_all, y_all, color='red')

#erkekler için linregress methoduyla m ve n değerlerine ulaşıyorum.
result_male = stats.linregress(a[a[:, 3] == 0, 1], a[a[:, 3] == 0, 2])
#print(result_male)

#erkekler için elde ettiğim korelasyon katsayılarını y=mx+n denklemimde yazıyorum
x_male = np.linspace(165, 200, 100)#165-200 arasında 100 tane eleman ataması yapıyorum
y_male = result_male.slope * x_male + result_male.intercept
plt.plot(x_male, y_male, color='blue')

#kadınlar için linregress methoduyla m ve n değerlerine ulaşıyorum.
result_female = stats.linregress(a[a[:, 3] == 1, 1], a[a[:, 3] == 1, 2])
#print(result_female)
# =============================================================================
# print(result)
# print()
# =============================================================================

#kadınlar için elde ettiğim korelasyon katsayılarını y=mx+n denklemimde yazıyorum
x_female = np.linspace(165, 200, 100)#165-200 arasında 100 tane eleman ataması yapıyorum
y_female = result_female.slope * x_female + result_female.intercept
plt.plot(x_female, y_female, color='green')
plt.title('All Graphs')
plt.pause(0.5)
plt.show()

#tahmin yapacağım veri setimi okuyorum veri setimde yalnızca boy parametremisini alıyorum
x_predict_data = np.loadtxt('prediction.txt', usecols=(2, ), delimiter=',', encoding='utf-8')

#boylarını okuduklarımın kilo kestiriminde bulunuyorum
y_predict = result_all.slope * x_predict_data * 100 + result_all.intercept

for i in range(len(x_predict_data)):
    print(f'{x_predict_data[i]} ---> {y_predict[i]}')


