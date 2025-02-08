import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # çıkan sonuçları görselleştirmek için kullanacağız

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV #GridSearchCV'ı knn ile ilgili best parametreleri bulurken kullanacağız
from sklearn.metrics import accuracy_score, confusion_matrix #Modelimizi değerlendirirken kullanacağız
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

#warning kaldırmak için
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("cancer.csv")
data.drop(['Unnamed: 32','id'],inplace = True,axis=1) # unnamed: 32 ve id adlı kolonları sil ve kendisine(data değişkenine) eşitle

data = data.rename(columns= {"diagnosis":"target"}) #diagnosis adlı kolon adını target yap

data["target"] = data["target"].astype("category")
sns.countplot(x="target", data=data) #iyi huylu ve kötü huylu sayılarını grafikleştir

print(data.target.value_counts()) #iyi huylu ve kötü huylu sayılarını yazdır (grafiğe göre daha net sayı yazar)

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target] #eğer M ise 1 yap B ise 0 yap (i.strip() boşlukları kaldırır)


print(len(data))
print(data.head())
print("Data Shape: ",data.shape)

data.info()

describe = data.describe()
"""
1-) %37 tane benign var
2-) %63 tane malign var

3 -) Veriler arası büyük bir scale farkı var(bazıları 0.37 bazıları 654 vs.) bu yüzden standarization gerekli.
Çünkü büyük sayılar küçük sayılara baskın gelebilir
4-) concavity_mean kolonunda 0 değerler var bunlar missing value mi? yoksa threshold yüzünden mi 0 oldu?
ama veri setini hazırlayanlar missing value olmadığını belirtti
missing value: none
"""

#%% EDA

# Correlation

corr_matrix = data.corr() # Sayısal veriler arasındaki kolerasyon'a bakacak eğer category(örneğin string veri) veri olsaydı discard ederdi.
sns.clustermap(corr_matrix,annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

"""
kısaca iki feature arasındaki correlation'a bakıyoruz bu değer 1 ise bu iki ilişki birbiriyle doğru orantılıdır
eğer -1 ise bu iki feature ters orantılı 0 ise herhangi bir ilişki yok
Bu ne işimize yarayacak?

Featureler birbirleriyle alakalıysa modelimize olan katkıları aynıdır
Bu nedenler machine learning modelimizde çeşitliliğe gitmeliyiz.
çeşitlik dediğimiz şey de birbirleri arasında ilişki olmayan featurelar (grafikte görülebilir, örneğin symetry_worst ile ..._dimension_se arasında 0.11 değeri var)

"""
# Sadece 0.75 değerinden yüksek ilişkilere bakalım
threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold # negatif değerler olduğu için mutlak değerini alıyoruz
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.5")

 

# Box plot
data_melted = pd.melt(data,id_vars = "target",
                      var_name = "features",
                      value_name= "value")

plt.figure()
sns.boxplot(x = "features", y = "value",hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

# box plottan herhangi bir anlam çıkaramadık çünkü veriler arası fazla scale var standarize/normalize ettiğimiz zaman düzelir.
# box plot şimdilik dursun




# pair plot
sns.pairplot(data[corr_features],diag_kind= "kde",markers="+",hue= "target")
plt.show()
"""
corelated featureler  5 tane feature 0.75ten yüksek değere sahip featurelardır
diag_kind ise diagonal görselleştirmemizin türevi
markers noktalar
hue bizim classlarımız (target)
2  tane class olduğu için görüntüde 2 renk var  
"""


#%% OUTLİER DETECTİON

"""
Outlier veri seti içerisinde bulunan aykırı değerlerdir. Eğer ayıklanmazsa modeli yanlış yönlendirebilir.
Densitiy base outlier detection system yöntemininin içerisinde lcoal outlier factor kullanacağız
neden bunu kullanırız çünkü bizim verimiz skew data bunun içerisindeki outlier'ları tespit etmek bu yöntemle
etkili oluyor 

compare local density of one point to
local density of K NN 

Bir k değeri seçiyoruz # genelde bu yöntemde k değeri 20 seçilir
örnekte basit olsun diye 2 seçelim

Resimdeki örnekte A noktasının bir outlier değer olup olmadığını hesaplarsak
bu A noktası için LOF(A) değer > 1 ise bu nokta outlier noktadır
eğer < 1 ise outlier nokta değildir. Bu 1 değeri değişebilir

Bu LOF(A) nasıl hesaplarnır
LOF(A) = ((LRDb + LRDd) / LRDa ) * (1/k)



LRD = Local Reachibility Density

LRD = 1/ARD
ARD = Average Reachibility Distance

RD = Reachibilty Distance

Neden LRDb ve LRDd dedik çünkü a noktasına en yakın 2 komşusuyla karşılaştırma yapacaktık (tanımda da belirtmiştik)


"""

y = data.target
x = data.drop(["target"],axis = 1)

columns = x.columns.tolist()


clf = LocalOutlierFactor() # n değeri deafult olarak 20 seçilmiştir
y_pred = clf.fit_predict(x) #outlier değerler için -1 döndürür inlier değerler için de 1 döndürür
X_score = clf.negative_outlier_factor_


outlier_score = pd.DataFrame()
outlier_score["score"] = X_score


"""
569 değerde yaklaşık 30 tane outlier çıkıyor bunları silmeye gerek yok veri kaybı yaşamayalım
Bir threshold belirleyip ona göre outlier değerleri çıkaralım Daha iyi görebilmek için görselleştirelim

"""

#threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color = "blue",s = 50,label = "Outliers")



plt.scatter(x.iloc[:,0],x.iloc[:,1],color = "k",s = 3, label = "Data Points")


radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1], s = 1000*radius ,edgecolors = "r",facecolors = "none",label = "Outlier Scores") #facecolors = noktacığın içerisinin rengi
plt.legend()
plt.show()


#drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values #ileride kullanacağımız için seri hale getirdik, type'nı değiştirdik


#%% TRAİN TEST SPLİT
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = test_size , random_state= 42)

#%% STANDARİZATİON


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train

# Box plot
data_melted = pd.melt(X_train_df,id_vars = "target",
                      var_name = "features",
                      value_name= "value")

plt.figure()
sns.boxplot(x = "features", y = "value",hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

"""
Bu plotta her bir feature'a ait dağılımları görebiliyoruz (farklı classlar için 0 ve 1)

Bu plotta aykırı değerleri de görebiliyoruz

Hangi feature'ların güzel ayrılabileceğini belirleyebiliriz. Feature extraction için
dağılımlarında çok farklılıkları olmayanları ayırmayız
"""

sns.pairplot(X_train_df[corr_features],diag_kind= "kde",markers="+",hue= "target")
plt.show()


#%% Basic KNN Method

"""
Training olmadığı için hızlı
implement edilmesi kolay
tune etmesi kolay


dezavantaj
knn outlier'a karşı sensitive'dir (outlier'ları temizlememiz gerek)
big data'da sıkıntılıdır
 
"""

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test,Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic Knn acc: ",acc)


"""
KNN sonucu % 95.3
"""


"""
CM:  [[108   1]
      [  7  55]]

109 iyi huylu'dan 108 tane doğru 1 tane yanlış tahmin etmiş
63 kötü huylu'dan 55 tane doğru tahmin etmiş 7 tane yanlış tahmin etmiş
"""

#%% Choose best paramters

def KNN_Best_Params(x_train,x_test,y_train,y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range,weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train,y_train)
    
    print("Best training Score: {} with parameters: {}".format(grid.best_score_,grid.best_params_))
    print()
    
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    
    acc_test = accuracy_score(y_test,y_pred_test)
    acc_train = accuracy_score(y_train,y_pred_train)
    
    print("Test Score: {}, Train Score: {}".format(acc_test,acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)


    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)


"""
KNN sonucu % 95.3
KNN Best paramters sonucu %95.9

"""


"""
train set error: %0
test set error: %6

ise overfitting var

train set error: %30
test set error: %31

ise underfitting var


train set error: %15
test set error: %30

ise overfitting var

cross validation yapılmalı(yapıldı) ve model complexity azaltılmalı 

Çapraz doğrulama ile overfitting ve underfitting riskleri değerlendirilmiş.

"""

#%% PCA

"""

PCA Mümkün olduğu kadar bilgi tutarak verinin boyutunun azaltılmasını sağlayan yöntemdir
neden PCA kullanırız belli başlı bir zaman kısıtımız varsa ve verinin boyutu çok fazlaysa (feature sayısı)
varsa PCA ile belli başlı feature'ları azaltabiliriz yani verinin boyutunu azaltabiliriz

Eğer elimizde bir kolerasyon matrisi varsa ve orada belli başlı feature'lar varsa bu feature'lar birbirleriyle
correlated feature'lar ise bu featureları nasıl çıkartacağımızı bilmiyorsak bu feature'ların ortadan kalkmasını
PCA yöntemini kullanabiliriz

Diğer bir kullanış amacı da görselleştirme
çok boyutlu verileri görselleştiremeyiz bu nedenle PCA yöntemi kullanılır
Çok boyutlu verileri az boyuta indirecez (30 -> 2)



Verileri ilk olarak 0 merkeze çekiyoruz
Sonrasında covaryans matrisi bulmamız gerek
Sonra eigenleri buluyoruz 

"""

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(X_reduced_pca,columns= ["p1","p2"])
pca_data["target"] = y

sns.scatterplot(x = "p1",y= "p2",hue = "target",data = pca_data)
plt.title("PCA: p1 vs p2")


"""
30 boyutlu veriyi 2 boyutta gösterdik
"""

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca,y,test_size = test_size , random_state= 42)


grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)    

"""
Hangi verilerin yanlış tahmin edildiğini görselleştirelim
"""

cmap_light = ListedColormap(['Orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])



h = .05
X = X_reduced_pca
x_min,x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min,y_max= X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))


Z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)


plt.scatter(X[:,0],X[:,1],c=y, cmap = cmap_bold,
            edgecolor='k',s = 20)


plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("%i-Class classification (k = %i, weights= '%s'"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors,grid_pca.best_estimator_.weights))


"""
KNN sonucu % 95.3
KNN Best paramters sonucu %95.9
PCA sonucu %92 (veriyi 2 boyuta düşürdük)


"""

#%% NCA

nca = NeighborhoodComponentsAnalysis(n_components= 2 , random_state= 42)
nca.fit(x_scaled,y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns= ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x="p1",y="p2",hue="target",data = nca_data)
plt.title("NCA: p1 vs p2")




X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca,y,test_size = test_size , random_state= 42)


grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)  



h = .2
X = X_reduced_nca
x_min,x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min,y_max= X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))


Z = grid_nca.predict(np.c_[xx.ravel(),yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)


plt.scatter(X[:,0],X[:,1],c=y, cmap = cmap_bold,
            edgecolor='k',s = 20)


plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("%i-Class classification (k = %i, weights= '%s'"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors,grid_nca.best_estimator_.weights))



"""
KNN sonucu % 95.3
KNN Best paramters sonucu %95.9
PCA sonucu %92 (veriyi 2 boyuta düşürdük)
NCA sonucu %99 (veriyi 2 boyuta düşürdük)

"""
#%% Modeli kaydetme


import pickle
# Modeli eğittikten sonra
with open('cancer_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

#%%
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Modeli yükleme
with open('cancer_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

class CancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Göğüs Kanseri Sınıflandırması")

        # Özelliklerin isimlerini tanımlayın
        self.features = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean", 
            "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        # Kullanıcı girdilerini oluştur
        self.inputs = []
        num_columns = 2  # Kaç sütun istiyorsanız burayı değiştirin
        for i, feature in enumerate(self.features):
            row = i // num_columns
            col = (i % num_columns) * 2  # Sütunları 2 birim kaydırıyoruz
            label = tk.Label(root, text=f"{feature}:")
            label.grid(row=row, column=col, padx=10, pady=5)
            entry = tk.Entry(root)
            entry.grid(row=row, column=col + 1, padx=10, pady=5)
            self.inputs.append(entry)

        # Tahmin butonu
        predict_button = tk.Button(root, text="Tahmin Yap", command=self.make_prediction)
        predict_button.grid(row=(len(self.features) // num_columns) + 1, column=0, columnspan=num_columns * 2, pady=10)

        # Sonuç etiketi
        self.result_label = tk.Label(root, text="", fg="blue")
        self.result_label.grid(row=(len(self.features) // num_columns) + 2, column=0, columnspan=num_columns * 2, pady=10)

    def make_prediction(self):
        try:
            input_data = [float(entry.get()) for entry in self.inputs]
            if len(input_data) != len(self.features):
                raise ValueError("Eksik veri")
            input_array = np.array(input_data).reshape(1, -1)
            
            prediction = knn_model.predict(input_array)
            result = "Malign (Kötü Huylu)" if prediction[0] == 1 else "Benign (İyi Huylu)"
            self.result_label.config(text=f"Tahmin Sonucu: {result}")
        except ValueError:
            messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayılar girin.")

# Ana Tkinter döngüsü
if __name__ == "__main__":
    root = tk.Tk()
    app = CancerPredictionApp(root)
    root.mainloop()

