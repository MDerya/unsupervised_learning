


#           İş Problemi
# Online ayakkabı mağazası olan FLO müşterilerini
# segmentlere ayırıp bu segmentlere göre pazarlama
# stratejileri belirlemek istiyor. Buna yönelik olarak
# müşterilerin davranışları tanımlanacak ve bu
# davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

#           Veri Seti Hikayesi
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
#                   12 Değişken       19.945 Gözlem
# master_id                         Eşsiz müşteri numarası
# order_channel                     Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel                En son alışverişin yapıldığı kanal
# first_order_date                  Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date                   Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online            Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline           Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online       Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline      Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online  Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12       Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


########################################
#       GÖREV 1: Veriyi Hazırlama
########################################
import seaborn as sns
import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering #birlestiriciClustering metodunu import ediyorum

# Adım1: flo_data_20K.csv verisini okutunuz
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df_ = pd.read_csv("flo_data_20k.csv")
df=df_.copy()
df.head()
df.shape
df.isnull().sum()
df.info()
df.describe().T

datetime = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[datetime] = df[datetime].apply(pd.to_datetime)



# Adım2:Müşterileri segmentlerken kullanacagınız degiskenleri seciniz.

# Not:Tenure(müşterinin yaşı), Recency(en son kac gün önce alısveriş yaptıgı) gibi yeni degiskenler olusturabilirsiniz.

"""#Her bir müşterinin toplam alışveriş sayısı
df["order_sum"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]

#Her bir müşterinin toplam alışveriş harcaması
df["customer_value_sum"]= df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
"""
# RFM metrikleri
#  *Recency   :yenilik (müsterinin yenilik ya da bizden en son ne zaman alısveris yaptı durumunu ifade etmektedir.)
#  *Frequency :sıklık (müsterinin yaptıgı toplam alısveris sayısı/islem sayısı)
#  *Monetary  :parasal deger (müsterilerin bize bıraktıgı parasal deger)
#  Tenure : müşterinin yaşı (bugünün tarihi - ilk alışveriş yaptıgı tarih)

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1) #analizin yapıldıgı tarih,
type(today_date)

#  Tenure : müşterinin yaşı
df["tenure"]= (df["last_order_date"] - df["first_order_date"]).dt.days #dt.days yapmazsak tipi datetime oluyor ve sonunda ...days yazıyor

df["recency"]= (today_date - df["last_order_date"]).dt.days

df.info()

######################################################
#       GÖREV 2: K-Means ile Müşteri Segmentasyonu
######################################################

# Adım 1: Degişkenleri standartlaştırınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#num_cols u kendim tanımlıyorum
num_cols=["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", "tenure", "recency"]
model_df= df[num_cols] #iki parantez önemli

#degiskenleri standartlaştırmadan önce çarpıklıga bakmamız gerek
#SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Normal dağılımın sağlanması için Log transformation uygulanması
model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()

#(k-means) uzaklık temelli bir yöntem kullanıcaz, uzaklık temelli ve gradient descent temelli yöntemlerin
#kullanımındaki süreclerde degiskenlerin standartlaştırılması önem arzediyor, dolayısıyla buradaki degiskenleri standartlastırmamız gerekiyor
#sc = MinMaxScaler((0, 1)) #ya da standart scaler tercih edilebilir
#model_df = sc.fit_transform(model_df)
#df[num_cols][0:5] #fit_transformdan cıktıktan sonra bunlar numpy arrayine dönüstü,dolayısıyla numpy array old icin df.head yapamıyoruz
#ya da
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()


# Adım 2: Optimum küme sayısını belirleyiniz.

#öyle bir işlem yapmalıyım ki, farklı k parametre degerlerine göre ssd(ssr,sse aynı seyler) leri inceleyip onlara göre karar vermeliyim
"""kmeans = KMeans() #boş bir Kmeans nesnesi olusturuyorum
ssd = [] #boş bir ssd listesi olusturuyorum
K = range(1, 30) #1den 30a kadar k lar olusturuyorum

for k in K: #bütün K larda gezip
    kmeans = KMeans(n_clusters=k).fit(model_df) #bu aralıktaki bütün k ları buraya girecek, fit edecek
    ssd.append(kmeans.inertia_) #sonra inertia degerlerini ssd nin icine gönderecek

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()"""
#ya da
#karar vermek icin egimin en siddetli oldugu noktalar secilir,bunun daha otomatik bir yolu var
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()
#grafikte bu veri setini kümelere ayırmak istiyorsan optimum nokta 6 dır demiş,grafikte maviler ssd ler,siyah cizgi de sectigi optimum nokta
elbow.elbow_value_

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=42).fit(model_df) #model kuruyoruz

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[num_cols][0:5]

clusters_kmeans = kmeans.labels_


final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]

final_df["kmeans_cluster"] = clusters_kmeans #yeni bir degisken ekledim, hangi cluster'lardan oldugu bilgisini girdim
final_df["kmeans_cluster"].unique()
final_df.head()

final_df["kmeans_cluster"] = final_df["kmeans_cluster"] + 1 #cluster'lar 0 dan degil de 1 den başlasın istiyorum

final_df[final_df["kmeans_cluster"]==6] #6 numaralı cluster da hangi müşterilerin olduguna bakalım




# Adım 4: Herbir segmenti istatistiksel olarak inceleyiniz.

final_df.head()
final_df.groupby("kmeans_cluster")["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", "tenure", "recency"].agg(["mean", "median","min","max","count", ])

######################################################
#       GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
######################################################

# Adım 1: Görev 2 de standartlaştırdıgınız dataframe i kullanarak optimum küme sayısını belirleyiniz.


# linkage yöntemi birlestirici clustering yöntemi
#öklid uzaklıgına göre gözlem birimlerini kümelere ayırıyor, daha sonra birbirine benzer olanları buluyor
hc_complete = linkage(model_df, 'complete')
"""* method='complete' assigns

        .. math::
           d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

      * method='average' assigns

        .. math::
           d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}
                                   {(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm."""

#dendogram : kümeleme yapısını gösteren şema
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

#küme sayısını belirlemek
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--') #cizgi atma işini y eksenindeki belirli bir degere göre yapıyoruz
plt.axhline(y=1, color='b', linestyle='--')
plt.show()

# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

#grafikten küme sayısını 6 olarak belirledim
hcluster = AgglomerativeClustering(n_clusters=6)
clusters = hcluster.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["hi_cluster_no"] = clusters  #hiyerarsik clusterno

final_df["hi_cluster_no"] = final_df["hi_cluster_no"] + 1
final_df["hi_cluster_no"].unique()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyiniz.



final_df.groupby("hi_cluster_no")["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", "tenure", "recency"].agg(["mean", "median","min", "max", "count"])
