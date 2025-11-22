# 1. Import Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.style.use('fivethirtyeight')

# 2. Import Data
df = pd.read_csv('Mall_Customers.csv')
df.head()

# 3. Amati bentuk data
df.shape

# 4. Ringkasan Statistik Deskriptif
df.describe(include='all')

# 5. Cek Missing Value
df.isnull().sum()

# 6. Cek Outlier (IQR method)
num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Outlier kolom {col}: {len(outliers)} data")

# 7. Visualisasi Distribusi Fitur
plt.figure(1, figsize=(15,6))
n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", bins=20)
    plt.title(f'Distplot of {x}')

plt.show()

# 8. Pairwise Relationship Plot (Regplot)
plt.figure(1, figsize=(15,20))
n = 0
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for x in cols:
    for y in cols:
        n += 1
        plt.subplot(3,3,n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha':0.6})
        plt.ylabel(y)

plt.show()

# 9. Scatter Gender vs Income & Score
plt.figure(1, figsize=(15,8))

for gender in ['Male', 'Female']:
    plt.scatter(
        x=df[df['Gender']==gender]['Annual Income (k$)'],
        y=df[df['Gender']==gender]['Spending Score (1-100)'],
        s=200, alpha=0.5, label=gender
    )

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income vs Spending Score by Gender')
plt.legend()
plt.show()

# 10. Elbow Method untuk menentukan jumlah cluster
X1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10,
                   max_iter=300, random_state=111)
    model.fit(X1)
    inertia.append(model.inertia_)

plt.figure(1, figsize=(15,6))
plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 11. Build K-Means (isi jumlah cluster sesuai elbow)
k_opt = 5   
model = KMeans(n_clusters=k_opt, init='k-means++', n_init=10,
               max_iter=300, random_state=111, algorithm='elkan')

model.fit(X1)
labels = model.labels_
centroids = model.cluster_centers_

# 12. Grid untuk visualisasi boundary cluster
step = 0.02
x_min, x_max = X1[:,0].min() - 1, X1[:,0].max() + 1
y_min, y_max = X1[:,1].min() - 1, X1[:,1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step),
    np.arange(y_min, y_max, step)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 13. Plot Cluster
plt.figure(1, figsize=(15,7))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

plt.scatter(X1[:,0], X1[:,1], c=labels, s=250)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=300, alpha=0.6)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering Result')
plt.show()

# 14. Silhouette Score
score = silhouette_score(X1, labels)
print("Silhouette Score:", score)
