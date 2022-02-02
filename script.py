import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import decomposition
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.utils import shuffle
import umap




class Digits:
    def __init__(self,name):
        self.name = name   
    
    def get_plots(self, dataset_name,n):
        if dataset_name == "digits":
            digits = load_digits()
            X = digits.data
            y = digits.target
            print(X.shape)
            pca = PCA(n) 
            X = pca.fit_transform(X)
            print(X.shape)
            plt.style.use('seaborn-whitegrid')
            plt.figure(figsize = (10,24))
            c_map = plt.cm.get_cmap('jet', 10)
            plt.scatter(X[:, 0], X[:, 1], s = 15,
                        cmap = c_map , c = y)
            plt.colorbar()
            plt.xlabel('PC-1') , plt.ylabel('PC-2')
            plt.show()
            
            #using umap for creating embedding
            reducer = umap.UMAP(random_state=42)
            reducer.fit(X)


            embedding = reducer.transform(X)
            # Verify that the result of calling transform is
            # idenitical to accessing the embedding_ attribute
            assert(np.all(embedding == reducer.embedding_))
            embedding.shape
            plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
            plt.title('UMAP projection of the Digits dataset', fontsize=24)
            plt.show()

            #plot of the same
            #utilising HDBSCAN for clustering
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(embedding)
            clusterer.labels_

            clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(embedding)
            color_palette = sns.color_palette('deep',50)
            cluster_colors = [color_palette[x] if x >= 0
                              else (0.5, 0.5, 0.5)
                              for x in clusterer.labels_]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, clusterer.probabilities_)]
            plt.scatter(*embedding.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
            plt.show()

            #score for representing the outliers
            score = clusterer.probabilities_

            mask_outliers = (score < 0.3)
            mask_inliers = ~mask_outliers
            plt.scatter(embedding[mask_inliers, 0], embedding[mask_inliers, 1], c= 'g', label='inliner')
            plt.scatter(embedding[mask_outliers, 0], embedding[mask_outliers, 1], c= 'r', label='outliers')
            plt.legend()
            plt.show()

            sorted_scores = np.argsort(score)

            #clustering using k-means

            X_kmeans = KMeans(n_clusters=3)
            X_kmeans.fit(embedding)
            # Predicting the cluster labels
            labels = X_kmeans.predict(embedding)
            print(labels)
            # Finding the final centroids
            centroids = X_kmeans.cluster_centers_

            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap = 'rainbow')
            plt.show()

            #not required but just a verification of K calculated using elbow method
            K = range(1,15)
            sum_of_squared_distances = []
            # Using Scikit Learn’s KMeans Algorithm to find sum of squared distances
            for k in K:
                model = KMeans(n_clusters=k).fit(embedding)
                sum_of_squared_distances.append(model.inertia_)
            plt.plot(K, sum_of_squared_distances, 'bx-')
            plt.xlabel('K values')
            plt.ylabel('Sum of Squared Distances')
            plt.title('Elbow Method')
            plt.show()


        
    
    
    
        


# In[8]:


d = Digits('digits')
d.get_plots('digits',24)


# In[14]:


class synthetic_data():
    def __init__(self,name):
        self.name = name   
    
    def get_plots(self, dataset_name,n):
        if dataset_name == "blobs":
            centers = [[1, 1], [-1, -1], [1, -1],[-1,1]]
            X, y = make_blobs(
                n_samples=600, n_features = 10, centers=centers, cluster_std=0.4, random_state=0
            )

            X = StandardScaler().fit_transform(X)
            print(X)

            #PCA to reduce dimenstions
            n_components = 2
            pca = decomposition.PCA(n_components=2)
            pc = pca.fit_transform(X)

            print(pc.shape)

            #simple plot to see variation distribution between PC1 and PC2
            df = pd.DataFrame({'var':pca.explained_variance_ratio_,
                         'PC':['PC1','PC2']})
            sns.barplot(x='PC',y="var", 
                       data=df, color="c");

            pc_df = pd.DataFrame(data = pc , 
                    columns = ['PC1', 'PC2'])
            pc_df['Cluster'] = y
            pc_df.head()

            #plot
            sns.lmplot( x="PC1", y="PC2",
              data=pc_df, 
              fit_reg=False, 
              hue='Cluster', # color by cluster
              legend=True,
              scatter_kws={"s": 80})

            #clustering using K-means
            

            X_kmeans = KMeans(n_clusters=3)
            X_kmeans.fit(X)
            # Predicting the cluster labels
            labels = X_kmeans.predict(X)
            print(labels)
            # Finding the final centroids
            centroids = X_kmeans.cluster_centers_

            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap = 'rainbow')
            plt.show()

            #verification of K using k-means
            K = range(1,10)
            sum_of_squared_distances = []
            # Using Scikit Learn’s KMeans Algorithm to find sum of squared distances
            for k in K:
                model = KMeans(n_clusters=k).fit(X)
                sum_of_squared_distances.append(model.inertia_)
            plt.plot(K, sum_of_squared_distances, 'bx-')
            plt.xlabel('K values')
            plt.ylabel('Sum of Squared Distances')
            plt.title('Elbow Method')
            plt.show()

            #clustering using dbscan
            db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)
            print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
            print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
            print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
            print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, labels))
            print(
                "Adjusted Mutual Information: %0.3f"
                % metrics.adjusted_mutual_info_score(y, labels)
            )
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

            # Plot result
            

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                )

            plt.title("Estimated number of clusters: %d" % n_clusters_)
            plt.show()

            #clustering using HDBSCAN
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(X)
            clusterer.labels_

            #clustering using HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(X)
            color_palette = sns.color_palette('deep', 8)
            cluster_colors = [color_palette[x] if x >= 0
                              else (0.5, 0.5, 0.5)
                              for x in clusterer.labels_]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, clusterer.probabilities_)]
            plt.scatter(*X.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

            score = clusterer.probabilities_

            #outlier demonstration
            mask_outliers = (score < 0.3)
            mask_inliers = ~mask_outliers
            plt.scatter(X[mask_inliers, 0], X[mask_inliers, 1], c= 'g', label='inliner')
            plt.scatter(X[mask_outliers, 0], X[mask_outliers, 1], c= 'r', label='outliers')
            plt.legend()

            sorted_scores = np.argsort(score)

            X[sorted_scores[1]]







m = synthetic_data('blobs')
m.get_plots('blobs',2)

