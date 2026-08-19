import pandas as pd
import numpy as np

from sklearn.metrics import adjusted_rand_score, silhouette_score
import igraph as ig
import leidenalg


class GlobalMapping():
    
    def __init__(self, df_distances, df_prob, df_total, distance_func, barycenter_func,
                 similarity_sigma=0.1, static_articles_threshold=500,
                 dyn_similarity_threshold=0.1, dyn_cluster_size_threshold=0.01):
        self._random_seed = 10
        self._year_list = df_prob.index.get_level_values(1).unique().sort_values().to_list()

        self._df_mapping = None
        self._df_barycenters = None

        self._silhouette_score = None
        self._ari_score_mean = None
        self._ari_score_std = None
        self._modularity = None

        self.detect_static_clusters(df_distances, df_total, similarity_sigma, static_articles_threshold)
        self.calculate_barycenters(df_prob)
        self.calculate_articles(df_total)
        self.detect_dynamic_clusters(
            distance_func,
            similarity_threshold=dyn_similarity_threshold,
            article_threshold=dyn_cluster_size_threshold)
        return

    @property
    def n_clusters(self):
        return

    @property
    def n_dynamic_clusters(self):
        return

    def n_clusters_by_year(self, year):
        return
    
    def n_dynamic_clusters_by_year(self, year):
        return
    
    @property
    def cluster_mapping(self):
        return

    def cluster_mapping_by_year(self, year):
        return

    @property
    def cluster_barycenters(self):
        return
    
    @property
    def dynamic_cluster_barycenters(self):
        return
    
    def cluster_barycenter_by_year(self, year):
        return
    
    def dynamic_cluster_barycenter_by_year(self, year):
        return

    def run_leiden_multiple(self, similarity_array, n_runs=10, seed_base=0):
        g = ig.Graph.Weighted_Adjacency(
            similarity_array.tolist(),
            mode="undirected",
            attr="weight",
            loops=False
        )

        partitions = []

        for i in range(n_runs):
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                seed=seed_base + i,
            )
            partitions.append(np.array(partition.membership))

        return np.array(partitions)  # shape: (n_runs, n_nodes)

    def compute_stability(self, partitions):
        n = len(partitions)
        scores = []

        for i in range(n):
            for j in range(i + 1, n):
                scores.append(adjusted_rand_score(partitions[i], partitions[j]))

        return np.mean(scores), np.std(scores)

    def build_coassignment_matrix(self, partitions):
        n_runs, n_nodes = partitions.shape
        coassoc = np.zeros((n_nodes, n_nodes))

        for p in partitions:
            coassoc += (p[:, None] == p[None, :]).astype(float)

        coassoc /= n_runs
        return coassoc
    
    def consensus_clustering(self, coassoc, seed=0):
        g = ig.Graph.Weighted_Adjacency(
            coassoc.tolist(),
            mode="undirected",
            attr="weight",
            loops=False
        )

        total_edge_weight = coassoc.sum().sum()

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=seed
        )

        return np.array(partition.membership), partition.quality() / (2 * total_edge_weight)

    def detect_static_clusters(self, df_distances, df_total, sigma, articles_threshold):
        df_static_mapping_list = []
        silhouette_list = []
        ari_mean_list = []
        ari_std_list = []
        modularity_list = []
    
        df_total_by_year = {year: df.droplevel(1) for year, df in df_total.groupby(level=1)}
        df_dist_by_year = {year: df.droplevel(1) for year, df in df_distances.groupby(level=1)}

        for i, year_loop in enumerate(self._year_list):
            
            country_list = df_total_by_year[year_loop].query("total >= @articles_threshold").index.to_list()
            df_dist_loop = df_dist_by_year[year_loop].loc[country_list, country_list]
            if isinstance(sigma, list):
                similarity_array = np.exp(- df_dist_loop.values ** 2 / (2 * sigma[i]**2))
            else:
                similarity_array = np.exp(- df_dist_loop.values ** 2 / (2 * sigma**2))

            partitions = self.run_leiden_multiple(similarity_array, n_runs=20)
            mean_ari, std_ari = self.compute_stability(partitions)

            coassoc = self.build_coassignment_matrix(partitions)
            labels, modularity = self.consensus_clustering(coassoc)

            silhouette_list.append(silhouette_score(df_dist_loop, labels, metric="precomputed"))
            modularity_list.append(modularity)
            ari_mean_list.append(mean_ari)
            ari_std_list.append(std_ari)

            df_static_mapping_list.append(pd.DataFrame({
                "country": country_list,
                "year": year_loop,
                "static_cluster": labels,
            }))

        self._silhouette_score = pd.DataFrame(silhouette_list, index=self._year_list, columns=["score"])
        self._modularity = modularity_list
        self._ari_score_mean = ari_mean_list
        self._ari_score_std = ari_std_list
        self._df_mapping = pd.concat(df_static_mapping_list)
        return
    
    def calculate_articles(self, df_total):
        self._df_articles_clusters = (
            self._df_mapping
            .merge(df_total, left_on=["country", "year"], right_index=True)
            .groupby(["year", "static_cluster"]).total.sum().to_frame("total")
        )

    def calculate_barycenters(self, df_prob):

        df_prob_flat = df_prob.reset_index()  # assumes (country, year) index
        df = self._df_mapping.merge(df_prob_flat, on=["country", "year"])
        # --- compute barycenters per (year, cluster) ---

        # print(df)
        df_barycenters = (
            df
            .drop(columns="country")
            .groupby(["year", "static_cluster"])
            .mean()
        )
        df_barycenters = df_barycenters.div(df_barycenters.sum(axis=1), axis=0)
        self._df_barycenters = df_barycenters
        return
    
    def detect_dynamic_clusters(self, distance_func, similarity_threshold=0.1, article_threshold=0.01, ):
        # similarity_threshold = 0.1
        # article_threshold = 0.01

        df_dyn_barycenters = pd.DataFrame({})

        df_mapping_tmp = self._df_mapping.copy()
        df_mapping_tmp["dynamic_cluster"] = df_mapping_tmp["static_cluster"]
        df_mapping_tmp = df_mapping_tmp.set_index(["year", "static_cluster"]).sort_index()

        for year in self._year_list:

            total = self._df_articles_clusters.xs(year)
            total_sum = total["total"].sum()

            mask = total["total"].to_numpy() >= (article_threshold * total_sum)
            unmask = total["total"].to_numpy() < (article_threshold * total_sum)
            passed_cluster = total.index[mask].to_list()
            not_passed_cluster = total.index[unmask].to_list()
            for cluster in not_passed_cluster:
                df_mapping_tmp.loc[(year, cluster), "dynamic_cluster"] = -1

            if len(passed_cluster) == 0:
                continue

            # --- barycenters ---
            idx = list(zip([year]*len(passed_cluster), passed_cluster))
            df_tmp = self._df_barycenters.loc[idx]

            # --- distance matrix ---
            if df_dyn_barycenters.empty:
                df_dyn_barycenters = df_tmp.xs(year)
                dyn_counts = len(df_tmp.index) - 1
                continue

            df_dist = distance_func(df_dyn_barycenters, df_tmp)

            dist_values = df_dist.to_numpy()  # shape: (dyn × clusters)

            dyn_index = df_dist.index.to_numpy()
            cluster_index = df_dist.columns.to_numpy()

            # --- greedy matching ---
            for j, cluster in enumerate(cluster_index):

                col_dist = dist_values[:, j]

                min_idx = col_dist.argmin()
                min_dist = col_dist[min_idx]
                best_dyn = dyn_index[min_idx]
                assigned = None
                if min_dist <= similarity_threshold:
                    assigned = best_dyn
                    df_dyn_barycenters.loc[assigned] = df_tmp.loc[cluster]
                else:
                    dyn_counts += 1
                    assigned = dyn_counts
                    df_dyn_barycenters.loc[assigned] = df_tmp.loc[cluster]
                df_mapping_tmp.loc[cluster, "dynamic_cluster"] = assigned

            self._df_mapping = df_mapping_tmp.reset_index()[["country", "year", "static_cluster", "dynamic_cluster"]]
            self._df_dyn_barycenters = (
                self._df_barycenters
                .merge(self._df_mapping.groupby(["year", "static_cluster"]).dynamic_cluster.mean().to_frame("dynamic_cluster").astype(int),
                    left_index=True, right_index=True)
                .reset_index()
                .set_index(["year", "dynamic_cluster"])
                .drop(columns=["static_cluster"])
            )
    
