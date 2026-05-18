from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# cell type separation
# --------------------------------------------------

def calculate_celltype_separation(adata):

    # leiden(?) ARI
    adjusted_rand_score()

    # Calinski-Harabasz
    calinski_harabasz_score()

    # Davies-Bouldin
    davies_bouldin_score()

    # silhouette score
    silhouette_score()



# --------------------------------------------------------
# predictability of neighbors (T-cells and tumors)
# --------------------------------------------------------

# are tumor marker genes in the top 20 highest weighted features for LR?

def lr_admixture_coefficients(adata,marker_genes,top_k=20):
    pass





# --------------------------------------------------
# similarity to reference
# --------------------------------------------------

# cosine similarity between clean/unclean and reference



