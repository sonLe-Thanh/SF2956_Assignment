import numpy as np
from gtda.time_series import SingleTakensEmbedding
from scipy.spatial.distance import cdist, pdist, squareform 
from readCSV import *

from stablerank import srank as sr

import matplotlib.pyplot as plt
from functools import reduce

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram





def extract_data_countries(file_path, exclude_list, num_countries, rescale=True, scale_factor=1e6, normalize=False):
    # Read the countries information and sort them
    list_country = readCSV(file_path, exclude_list)
    sorted_list = sort_country(list_country)

    list_top_countries = get_top_k(sorted_list, num_countries)
    # Preprocess the data of the countries
    for i in range(0, len(list_top_countries)):
        list_top_countries[i].remove_nan_heads()    
        list_top_countries[i].fill_missing_cells()
        if rescale:
            list_top_countries[i].rescale_data(scale_factor)
        if normalize:
            list_top_countries[i].normalize_data()

    return list_top_countries

def extract_stable_rank_countries(list_top_countries, clustering_method, auto_search_param=False,
                                embedding_dim=3, embedding_time_delay=1, stride=1,
                                max_embedding_dim=5, max_time_delay=2):
    
    # Preparation for the takens embedding
    # Using giotto-tda function
    # The value of the 2 parameters can be changed
    # Here we can have them manually or automatically 

    if auto_search_param:
        embedder = SingleTakensEmbedding(
            parameters_type="search",
            time_delay=max_time_delay,
            dimension=max_embedding_dim,
            stride=stride,
        )
    else:
        embedder = SingleTakensEmbedding(
            parameters_type="fixed",
            time_delay=embedding_time_delay,
            dimension=embedding_dim,
            stride=stride
        )

    # Transform the time-series to point clouds

    list_h0_sr = []
    list_h1_sr = []

    for i in range(len(list_top_countries)):
        country_val_point_cloud = embedder.fit_transform(list_top_countries[i].data)

        # Build the distance matrix
        distance_matrix = squareform(pdist(country_val_point_cloud))

        # getting a distance object
        c_dist = sr.Distance(distance_matrix)

        # getting H0 Stable rank
        sr_h0 = c_dist.get_h0sr(clustering_method = clustering_method)

        # Get H1 Stable rank
        barcodes = c_dist.get_bc(maxdim=2)

        sr_h1 = sr.bc_to_sr(barcodes, degree="H1")


        list_h0_sr.append(sr_h0)
        list_h1_sr.append(sr_h1)

    return list_h0_sr, list_h1_sr


def contruct_distance_mat_stable_rank(list_stable_rank):
    num_entries = len(list_stable_rank)
    # Now compute the distance between the stable ranks
    distance_mat = np.zeros((num_entries, num_entries))

    for i in range(num_entries):
        for j in range(num_entries):
            distance_mat[i, j] = list_stable_rank[i].interleaving_distance(list_stable_rank[j])
    
    return distance_mat


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def cluster_and_visualize(data, clustering_method, labels_list, color_threshold):
    clusterer_hier = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric="precomputed", linkage=clustering_method)
    clusterer_hier = clusterer_hier.fit(data)

    fig = plt.figure(figsize=(15, 10))
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(clusterer_hier, truncate_mode="level", p=50, labels=labels_list, orientation="right", color_threshold=color_threshold)
    plt.xlabel("Distance")
    
    return fig






    
# List of exclude name of countries
exclude_list = [
    "Caribbean small states",
    "East Asia & Pacific (excluding high income)",
    "Early-demographic dividend",
    "East Asia & Pacific",
    "Europe & Central Asia (excluding high income)",
    "Europe & Central Asia",
    "Fragile and conflict affected situations",
    "Heavily indebted poor countries (HIPC)",
    "IBRD only",
    "IDA & IBRD total",
    "IDA total",
    "IDA blend",
    "IDA only",
    "Latin America & Caribbean (excluding high income)",
    "Latin America & Caribbean",
    "Least developed countries: UN classification",
    "Low income",
    "Lower middle income",
    "Low & middle income",
    "Late-demographic dividend",
    "Middle income",
    "Middle East & North Africa (excluding high income)",
    "OECD members",
    "Other small states",
    "Pre-demographic dividend",
    "Pacific island small states",
    "Post-demographic dividend",
    "Sub-Saharan Africa (excluding high income)",
    "Sub-Saharan Africa",
    "East Asia & Pacific (IDA & IBRD countries)",
    "Europe & Central Asia (IDA & IBRD countries)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Middle East & North Africa (IDA & IBRD countries)",
    "South Asia (IDA & IBRD)",
    "Sub-Saharan Africa (IDA & IBRD countries)",
    "Upper middle income",
    "World",
    "High income",
    "Euro area",
    "European Union",
    "Africa Western and Central",
    "Arab World",
    "Middle East & North Africa",
    "Central Europe and the Baltics",
    "Africa Eastern and Southern",
    "Small states",
    "North America",
    "South America",
    "South Asia"
]

folder_path = "Dataset/"
file_path = "NumberOfArrivals.csv"

fig_folder = "results/avg_avg/dim3_delay1/NumArrivals/"
# Number of chosen countries
num_countries = 30
# Clustering methods for the stable rank
rank_clustering_method = "average"
# Clustering methods for the hierrachical clustering
hier_clustering_method = "average"
rescale = True

# Different embedding dim can have slightly different results, but the overall clustering results does not differ much
# We choose a embedding delay of 1 since our data for most of the time is heavily truncated after the preprocessing
# The embedding is 3
# These values are to balance between the quality and the quantity of our data
# Also, we won't work with receipts percentage, as our pipeline here can not properly handle it
# We also work with each file independently, as the top 30 of each files is different from the others
# Combo simple_complete is different, while the other 3 are almost identical
# We only consider the combo complete_st and simple_st here/ consider avg_avg 
# Ward is not available for precompute distance
embedding_dim = 3
embedding_delay = 1

list_countries = extract_data_countries(folder_path + file_path, exclude_list, num_countries, rescale=rescale, scale_factor=1e7, normalize=False)

label_list = []
for i in range(num_countries):
    label_list.append(list_countries[i].name)

list_stable_rank_h0, list_stable_rank_h1 = extract_stable_rank_countries(list_countries, clustering_method=rank_clustering_method,
                                                                            auto_search_param=False, embedding_dim=embedding_dim, embedding_time_delay=embedding_delay,
                                                                            stride=4, max_embedding_dim=5, max_time_delay=5)

# The average stable rank
avg_h0_sr = reduce(lambda x, y: x + y, list_stable_rank_h0[1:], list_stable_rank_h0[0]) / len(list_stable_rank_h0)
avg_h1_sr = reduce(lambda x, y: x + y, list_stable_rank_h1[1:], list_stable_rank_h1[0]) / len(list_stable_rank_h1)

dist_mat_rank_h0 = contruct_distance_mat_stable_rank(list_stable_rank_h0)
dist_mat_rank_h1 = contruct_distance_mat_stable_rank(list_stable_rank_h1)

# Clustering using DBSCAN
# DBSCAN marks a lot of countries as noise, not good

# Do the hierrachical clustering instead
clusterer_hier = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric="precomputed", linkage="single")


# Visualize the stable rank


fig_stable_rank_h0 = plt.figure("Rank H0", figsize=(15, 10))
for i in range(len(list_stable_rank_h0)):
    list_stable_rank_h0[i].plot(label=label_list[i])

plt.legend(ncol=2)
plt.xlabel("Distance", fontsize=20)
plt.ylabel("Rank", fontsize=20)
plt.title("H0 Stable rank of " + file_path, fontsize=25)
# Set tick font size for both axes
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

fig_stable_rank_h0.savefig(fig_folder+"H0.pdf", dpi=300)

# Draw with average
fig_stable_rank_h0_avg = plt.figure("Rank H0", figsize=(15, 10))
for i in range(len(list_stable_rank_h0)):
    list_stable_rank_h0[i].plot(label=label_list[i], alpha=0.6)

avg_h0_sr.plot(color="red", linewidth=2.5, label="Average")

plt.legend(ncol=2)
plt.xlabel("Distance", fontsize=20)
plt.ylabel("Rank", fontsize=20)
plt.title("H0 Stable rank of " + file_path + " with average", fontsize=25)
# Set tick font size for both axes
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

fig_stable_rank_h0_avg.savefig(fig_folder+"H0_avg.pdf", dpi=300)



# Draw the H1
fig_stable_rank_h1 = plt.figure("Rank H1", figsize=(15, 10))
for i in range(len(list_stable_rank_h1)):
    list_stable_rank_h1[i].plot(label=label_list[i])

plt.legend(ncol=2)
plt.xlabel("Distance", fontsize=20)
plt.ylabel("Rank", fontsize=20)
plt.title("H1 Stable rank of " + file_path, fontsize=25)
# Set tick font size for both axes
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

fig_stable_rank_h1.savefig(fig_folder+"H1.pdf", dpi=300)


# Draw with average
fig_stable_rank_h1_avg = plt.figure("Rank H0", figsize=(15, 10))
for i in range(len(list_stable_rank_h0)):
    list_stable_rank_h1[i].plot(label=label_list[i], alpha=0.6)

avg_h1_sr.plot(color="red", linewidth=2.5, label="Average")

plt.legend(ncol=2)
plt.xlabel("Distance", fontsize=20)
plt.ylabel("Rank", fontsize=20)
plt.title("H1 Stable rank of " + file_path + " with average", fontsize=25)
# Set tick font size for both axes
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

fig_stable_rank_h0_avg.savefig(fig_folder+"H1_avg.pdf", dpi=300)

# Visualize the clustering 

fig_cluster_h0 = cluster_and_visualize(dist_mat_rank_h0, hier_clustering_method, label_list, np.inf)
plt.xlabel("Distance", fontsize=20)
# plt.ylabel(fontsize=20)
plt.title("Clustering using H0 stable rank", fontsize=25)
plt.show()

fig_cluster_h0.savefig(fig_folder + "cluster_H0.pdf", dpi=300)


# Draw the H1
fig_cluster_h1 = cluster_and_visualize(dist_mat_rank_h1, hier_clustering_method, label_list, np.inf)
plt.xlabel("Distance", fontsize=20)
# plt.ylabel(fontsize=20)
plt.title("Clustering using H1 stable rank", fontsize=25)
plt.show()

fig_cluster_h1.savefig(fig_folder + "cluster_H1.pdf", dpi=300)