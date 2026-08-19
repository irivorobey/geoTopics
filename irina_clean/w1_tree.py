import pandas as pd
import numpy as np
from typing import Optional, List

np.seterr(divide='ignore', over='ignore', invalid='ignore')

class W1Tree():

    def __init__(
            self,
            df_tree: pd.DataFrame,
            level_columns: Optional[List[str]] = None,
            weight_list: Optional[List[float]] = None,
            weight_base = 10.0,
            norm: float = 1
        ):

        """
        Initialize the W1TreeDistance object.
        This class computes Wasserstein-1 (Earth Mover's) distances between 
        distributions supported on the leaves of a hierarchical tree.

        Parameters
        ----------
        df_tree : pd.DataFrame
            DataFrame describing the hierarchical tree structure. Each column 
            corresponds to a level in the hierarchy, and each row represents 
            a path from the root to a leaf.

        level_columns : list of str, optional
            Ordered list of column names defining the hierarchy levels.
            If None, all columns of `df_tree` are used.

        weight_list : list of float, optional
            Weights associated with each level of the tree. Must have the same 
            length as the number of levels. If None, default exponentially 
            decaying weights are used.

        norm : optional
            Maximum distance on a tree.

        Notes
        -----

        - The tree is augmented with an artificial root node.
        - The adjacency and distance matrices are precomputed for efficiency.

        """

        df_tree_root = df_tree.drop_duplicates().copy()
        df_tree_root.insert(0, "root", 0)
        self._df_tree = df_tree_root

        self._level_columns = level_columns
        self._n_by_levels = None
        self._n_leaves = len(df_tree_root.index)
        self._df_tree = df_tree_root[self.level_columns].sort_values(self.level_columns)

        if weight_list is None:
            a = norm / (2 * (1 / (weight_base ** np.arange(len(self._level_columns))))[1:].sum())
            self._weight_list = a * (weight_base ** np.arange(0, -len(self._level_columns), -1))
            # self._weight_list = a * (weight_base ** np.arange(0, len(self._level_columns), 1))
        else:
            # TODO: check size (the same as level_list)
            self._weight_list = weight_list
    
        self._adjacency_matrix = self.get_tree_adjacency_matrix()
        self.__distance_matrix = self.__get_distance_matrix()
        
        return

    @property
    def distance_matrix(self):
        """
        Wasserstein distance transformation matrix.

        Returns
        -------
        np.ndarray
            Matrix used to transform differences of leaf distributions into 
            Wasserstein distances.
        """
        return self.__distance_matrix

    @property
    def level_columns(self):
        """
        List of columns defining the hierarchical levels.

        Returns
        -------
        list of str
            Ordered list of level column names.
        """
        if self._level_columns is None:
            self._level_columns = self._df_tree.columns.tolist()
        return self._level_columns

    @property
    def ordered_leaves(self):
        """
        Ordered list of leaf nodes.

        Returns
        -------
        list
            Leaf identifiers corresponding to the last level of the hierarchy,
            sorted according to the internal tree representation.
        """
        return self._df_tree[self.level_columns[-1]].to_list()

    @property
    def n_by_levels(self):
        """
        Number of unique nodes at each level of the tree.
        
        Returns
        -------
        np.ndarray
            Array where each entry corresponds to the number of unique nodes 
            at a given level.
        """
        if self._n_by_levels is None:
            n_by_levels = []
            for level in self.level_columns:
                n_by_levels.append(self._df_tree[level].nunique())
            self._n_by_levels = np.array(n_by_levels)
        return self._n_by_levels

    def get_tree_adjacency_matrix(
            self,
        ) -> np.ndarray:
        """
        Construct the adjacency matrix of the hierarchical tree.
        The adjacency matrix encodes directed edges from parent nodes to 
        child nodes across consecutive levels.

        Returns
        -------
        np.ndarray
            A square binary matrix of shape (n_nodes, n_nodes), where entry 
            (i, j) is 1 if there is an edge from node i to node j, and 0 otherwise.

        Notes
        -----
        - Nodes are indexed in the order they appear across levels.
        - Only edges between consecutive levels are considered.
        """
        
        # Build mapping from node values to indices
        node_to_idx = {}
        idx_counter = 0
        
        for level in self.level_columns:
            for node in self._df_tree[level].unique():
                if node not in node_to_idx:
                    node_to_idx[node] = idx_counter
                    idx_counter += 1
        
        adj_matrix_size = len(node_to_idx)
        adj_matrix = np.zeros((adj_matrix_size, adj_matrix_size), dtype=np.int8)
        
        # Build edges between consecutive levels
        prev_level = self.level_columns[0]
        
        for level in self.level_columns[1:]:
            # Get unique parent-child relationships
            edges = self._df_tree[[prev_level, level]].drop_duplicates()
            
            # Vectorized assignment
            parent_indices = edges[prev_level].map(node_to_idx).values
            child_indices = edges[level].map(node_to_idx).values
            adj_matrix[parent_indices, child_indices] = 1
            
            prev_level = level
        
        return adj_matrix

    @property
    def weight_array(self):
        """
        Expanded weight vector aligned with all tree nodes.
        Each level weight is repeated according to the number of nodes 
        at that level.

        Returns
        -------

        np.ndarray
            One-dimensional array of weights corresponding to all nodes 
            in the tree.
        """
        weights = []
        n_by_levels = self.n_by_levels
        for i in range(len(self._n_by_levels)):
            weights += [self._weight_list[i]] * n_by_levels[i]
        weights_array = np.array(weights).T
        return weights_array

    def __get_distance_matrix(
            self
        ):
        """
        Compute the Wasserstein distance transformation matrix.
        This matrix encodes the contribution of each node in the tree to 
        the Wasserstein-1 distance between distributions on the leaves.

        Returns
        -------
        np.ndarray
            Matrix used to transform differences of leaf distributions into 
            Wasserstein distances.

        Notes
        -----
        - The computation is based on a block decomposition of the adjacency matrix.
        - Internal nodes and leaves are treated separately.
        - Matrix inversion is used; numerical issues may arise if the matrix 
        is ill-conditioned.
        """
        adj_matrix = self._adjacency_matrix
        
        n_nodes = self._adjacency_matrix.shape[0]
        n_leaves = self._n_leaves
        n_internal = n_nodes - n_leaves 

        adj_internal = adj_matrix[: n_internal, : n_internal]
        adj_leaves = adj_matrix[: n_internal, n_internal:]

        q11 = np.linalg.inv(np.eye(n_internal) - adj_internal)
        q12 = np.matmul(q11, adj_leaves)
        q21 = np.zeros((n_leaves, n_internal))
        q22 = np.eye(q12.shape[1])

        r1 = np.concatenate([q11, q12], axis=1)
        r2 = np.concatenate([q21, q22], axis=1)
        wass_array = np.concatenate([r1, r2])
        return np.tile(self.weight_array, (wass_array.shape[1], 1)).T * wass_array

    def dist(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Compute pairwise Wasserstein-1 distances between two sets of distributions.

        Parameters
        ----------

        df1 : pd.DataFrame
            First set of distributions. Rows correspond to observations and 
            columns correspond to leaf nodes.

        df2 : pd.DataFrame
            Second set of distributions, with the same structure as `df1`.

        Returns
        -------
        pd.DataFrame
            DataFrame of pairwise distances with rows indexed by `df1` and 
            columns indexed by `df2`.

        Notes
        -----
        - Input dataframes are aligned to the tree leaves; missing values are 
        filled with zeros.

        - The computation is fully vectorized for efficiency.

        - The result is reshaped into a pivot table format.
        """
        df1_aligned = df1.T.reindex(self.ordered_leaves).T.fillna(0)
        df2_aligned = df2.T.reindex(self.ordered_leaves).T.fillna(0)

        index1 = df1_aligned.index.to_list()
        index2 = df2_aligned.index.to_list()
        iterables = [index1, index2]
        m_index = pd.MultiIndex.from_product(iterables, names=["first", "second"])

        df_cross = (
            df1_aligned
            .merge(df2_aligned, how="cross")
        )

        diff_array = (df_cross.values[:, :len(self.ordered_leaves)] - df_cross.values[:, -len(self.ordered_leaves):])
        dist_array = np.sum((np.abs(self.__distance_matrix[:, -self._n_leaves:] @ diff_array.T)), axis=0)
        return pd.DataFrame(dist_array,
                            index=m_index,
                            columns=["distance"]).reset_index().pivot(index="first", columns="second").droplevel(0, axis=1)

    def project_to_simplex(self, v):
        """
        Euclidean projection of v onto the probability simplex:
            {x : x >= 0, sum(x) = 1}
        """
        v = v.flatten()
        n = len(v)
        u = np.sort(v)[::-1]  # sort descending
        cssv = np.cumsum(u)

        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)

        w = np.maximum(v - theta, 0)
        return np.array([w]).T

    def barycenter(self,
                   df: pd.DataFrame,
                   init_step_size: Optional[float] = 0.6,
                   step_size_decay: Optional[float] = 0.2,
                   max_iter: Optional[int] = 1000
                   ):
        """
        Compute the Wasserstein-1 barycenter of a set of distributions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of distributions with rows corresponding to observations 
            and columns corresponding to leaf nodes.

        Returns
        -------
        pd.Series
            Series representing the barycenter distribution over the leaves.

        Notes
        -----
        - The input dataframe is aligned to the tree leaves; missing values are 
        filled with zeros.

        - The barycenter is computed as the mean of the transformed distributions 
        in the Wasserstein space.
        """
        df_aligned = df.T.reindex(self.ordered_leaves).T

        n_distributions = df_aligned.index.size
        if n_distributions == 1:
            return df_aligned

        mask = df_aligned.notna().sum(axis=0).astype(bool).values
        mask_long = np.concatenate([np.array([True] * (self._n_by_levels.sum() - self._n_leaves)), mask])
        
        df_aligned = df_aligned.dropna(how="all", axis=1).fillna(0)

        b_matrix = self.__distance_matrix[:, -self._n_leaves:][mask_long, :][:, mask]
        b_vector = b_matrix @ df_aligned.values.T

        # a_step = np.ones((mask.sum(), 1)) / mask.sum()
        # a_step = np.array([df_aligned.iloc[1].values]).T
        a_step = np.mean(df_aligned.values, axis=0).reshape(-1, 1)
        a_step = self.project_to_simplex(a_step)
        a_best = a_step

        f_step_func = lambda a: np.mean(np.sum(np.abs(b_matrix @ a - b_vector), axis=0))
        g_step_func = lambda a: (np.sum(b_matrix.T @ np.sign(b_matrix @ a - b_vector), axis=1)) / n_distributions

        a_best = a_step
        f_best = f_step_func(a_step)
        if np.linalg.norm(g_step_func(a_step)) == 0:
            df_best = pd.DataFrame(a_best.T, columns=df_aligned.columns, index=["barycenter"])
            df_best = df_best.T.reindex(self.ordered_leaves).T
            return df_best
        # print(np.linalg.norm(g_step_func(a_step)), 1 / np.linalg.norm(g_step_func(a_step)))

        for i in range(max_iter):
            g_step = g_step_func(a_step)
            if g_step.sum() == 0:
                break
            step_size = init_step_size / (i + 1)**step_size_decay / np.linalg.norm(g_step)
            a_step = a_step - np.array([step_size * g_step]).T
            a_step = self.project_to_simplex(a_step)

            if f_best > f_step_func(a_step):
                f_best = f_step_func(a_step)
                a_best = a_step
        df_best = pd.DataFrame(a_best.T, columns=df_aligned.columns, index=["barycenter"])
        df_best = df_best.T.reindex(self.ordered_leaves).T
        return df_best
        