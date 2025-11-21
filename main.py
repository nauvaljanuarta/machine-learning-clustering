# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')

# 1. Load and inspect data
df = pd.read_csv('student-mat.csv')
print("Data Head:")
print(df.head())
print("\nData Shape:", df.shape)

# 2. Data information
print("\nData Info:")
print(df.info())

# 3. Check for null values
print("\nNull Values:")
print(df.isnull().sum())

# 4. Prepare categorical data for K-Modes
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                      'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                      'nursery', 'higher', 'internet', 'romantic']

categorical_data = df[categorical_columns].copy()

print("\nCategorical Data Head:")
print(categorical_data.head())
print("\nCategorical Data Description:")
print(categorical_data.describe())

# 5. Check unique values for each categorical column
print("\nUnique values in each categorical column:")
for column in categorical_columns:
    print(f"{column}: {categorical_data[column].unique()}")

# 6. Visualize categorical data distribution
plt.figure(figsize=(20, 15))

for i, column in enumerate(categorical_columns[:12], 1):  # First 12 columns
    plt.subplot(3, 4, i)
    sns.countplot(data=categorical_data, x=column)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 7. Continue visualization for remaining columns
plt.figure(figsize=(20, 10))

for i, column in enumerate(categorical_columns[12:], 1):  # Remaining columns
    plt.subplot(2, 3, i)
    sns.countplot(data=categorical_data, x=column)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 8. Determine optimal number of clusters using Elbow Method for K-Modes
cost = []
K = range(1, 8)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init="Cao", n_init=5, verbose=0, random_state=42)
    kmode.fit(categorical_data)
    cost.append(kmode.cost_)

plt.figure(figsize=(10, 6))
plt.plot(K, cost, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Method for Optimal Number of Clusters (K-Modes)')
plt.grid(True)
plt.show()

# 9. Apply K-Modes clustering with optimal number of clusters
optimal_clusters = 4  # Based on elbow method observation
kmode = KModes(n_clusters=optimal_clusters, init="Cao", n_init=5, verbose=1, random_state=42)
clusters = kmode.fit_predict(categorical_data)

# Add cluster labels to original data
categorical_data['Cluster'] = clusters
df['KMode_Cluster'] = clusters

print(f"\nK-Modes clustering completed with {optimal_clusters} clusters")

# 10. Display cluster centers (modes)
print("\nCluster Centers (Modes):")
for i, center in enumerate(kmode.cluster_centroids_):
    print(f"\nCluster {i}:")
    for j, feature in enumerate(categorical_columns):
        print(f"  {feature}: {center[j]}")

# 11. Analyze cluster composition
print("\nCluster Sizes:")
cluster_sizes = categorical_data['Cluster'].value_counts().sort_index()
print(cluster_sizes)

# 12. Visualize cluster distribution
plt.figure(figsize=(15, 10))

# Plot 1: Cluster sizes
plt.subplot(2, 3, 1)
plt.bar(cluster_sizes.index, cluster_sizes.values)
plt.title('Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Students')

# Plot 2: School distribution per cluster
plt.subplot(2, 3, 2)
school_cluster = pd.crosstab(categorical_data['Cluster'], categorical_data['school'])
school_cluster.plot(kind='bar', ax=plt.gca())
plt.title('School Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Plot 3: Gender distribution per cluster
plt.subplot(2, 3, 3)
gender_cluster = pd.crosstab(categorical_data['Cluster'], categorical_data['sex'])
gender_cluster.plot(kind='bar', ax=plt.gca())
plt.title('Gender Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Plot 4: Address distribution per cluster
plt.subplot(2, 3, 4)
address_cluster = pd.crosstab(categorical_data['Cluster'], categorical_data['address'])
address_cluster.plot(kind='bar', ax=plt.gca())
plt.title('Address Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Plot 5: Higher education aspiration per cluster
plt.subplot(2, 3, 5)
higher_cluster = pd.crosstab(categorical_data['Cluster'], categorical_data['higher'])
higher_cluster.plot(kind='bar', ax=plt.gca())
plt.title('Higher Education Aspiration per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Plot 6: Internet access per cluster
plt.subplot(2, 3, 6)
internet_cluster = pd.crosstab(categorical_data['Cluster'], categorical_data['internet'])
internet_cluster.plot(kind='bar', ax=plt.gca())
plt.title('Internet Access per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# 13. Detailed analysis of key features across clusters
key_features = ['school', 'sex', 'address', 'Mjob', 'Fjob', 'higher', 'internet']

print("\nDetailed Cluster Analysis:")
for feature in key_features:
    print(f"\n{feature} distribution across clusters:")
    cross_tab = pd.crosstab(categorical_data['Cluster'], categorical_data[feature], normalize='index') * 100
    print(cross_tab.round(2))

# 14. Analyze relationship between clusters and numerical features (for context)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='KMode_Cluster', y='age')
plt.title('Age Distribution per Cluster')

plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='KMode_Cluster', y='G3')
plt.title('Final Grade (G3) Distribution per Cluster')

plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='KMode_Cluster', y='absences')
plt.title('Absences Distribution per Cluster')

plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='KMode_Cluster', y='studytime')
plt.title('Study Time Distribution per Cluster')

plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='KMode_Cluster', y='failures')
plt.title('Failures Distribution per Cluster')

plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='KMode_Cluster', y='Medu')
plt.title("Mother's Education Distribution per Cluster")

plt.tight_layout()
plt.show()

# 15. Create cluster profiles
print("\n=== CLUSTER PROFILES ===")
for cluster in sorted(df['KMode_Cluster'].unique()):
    cluster_data = df[df['KMode_Cluster'] == cluster]
    cat_cluster_data = categorical_data[categorical_data['Cluster'] == cluster]
    
    print(f"\n--- CLUSTER {cluster} (n={len(cluster_data)}) ---")
    
    print("Demographic Profile:")
    print(f"  School: {cat_cluster_data['school'].mode()[0]} ({cat_cluster_data['school'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    print(f"  Gender: {cat_cluster_data['sex'].mode()[0]} ({cat_cluster_data['sex'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    print(f"  Address: {cat_cluster_data['address'].mode()[0]} ({cat_cluster_data['address'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    
    print("Family Background:")
    print(f"  Mother's Job: {cat_cluster_data['Mjob'].mode()[0]}")
    print(f"  Father's Job: {cat_cluster_data['Fjob'].mode()[0]}")
    print(f"  Family Size: {cat_cluster_data['famsize'].mode()[0]}")
    
    print("Educational Aspects:")
    print(f"  Wants Higher Education: {cat_cluster_data['higher'].mode()[0]} ({cat_cluster_data['higher'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    print(f"  Internet Access: {cat_cluster_data['internet'].mode()[0]} ({cat_cluster_data['internet'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    print(f"  Extra Paid Classes: {cat_cluster_data['paid'].mode()[0]} ({cat_cluster_data['paid'].value_counts(normalize=True).iloc[0]*100:.1f}%)")
    
    print("Academic Performance (Averages):")
    print(f"  Final Grade (G3): {cluster_data['G3'].mean():.2f}")
    print(f"  Age: {cluster_data['age'].mean():.2f}")
    print(f"  Absences: {cluster_data['absences'].mean():.2f}")

# 16. Save the results
df.to_csv('student_math_kmodes_results.csv', index=False)
categorical_data.to_csv('categorical_data_clusters.csv', index=False)

print("\n" + "="*50)
print("K-MODES CLUSTERING SUMMARY")
print("="*50)
print(f"Total students clustered: {len(df)}")
print(f"Optimal number of clusters: {optimal_clusters}")
print(f"Cluster sizes: {dict(cluster_sizes)}")
print(f"\nResults saved to:")
print("- student_math_kmodes_results.csv")
print("- categorical_data_clusters.csv")
print("\nClustering completed successfully!")