
import pandas as pd
import numpy as np

class SpecificityMetric():

    def __init__(self, df_prob):
        self._df_prob = df_prob
        self.df_metric = None
        self._pos_metric = None
        self._pos_metric_norm = None
        return

    @property
    def levels(self):
        return self._df_prob.index.names
    
    def get_metric(self, level_agg=0, level_norm=None):
        if level_norm is None:
            df_norm = self._df_prob.copy()
            df_norm_var = self._df_prob.var()
            df_n_norm = self._df_prob.sum(axis=1).count()
            level_agg_name = self.levels[level_agg]
            df_agg = self._df_prob.groupby(level=level_agg).mean()
            df_n_agg = self._df_prob.sum(axis=1).groupby(level=level_agg).count()
        else:
            level_norm_name = self.levels[level_norm]
            df_norm = self._df_prob.groupby(level=level_norm).mean()
            df_n_norm = self._df_prob.sum(axis=1).groupby(level=level_norm).count()
        
            level_agg_name = self.levels[level_agg]
            df_agg = self._df_prob.groupby(level=[level_norm, level_agg]).mean()
            df_n_agg = self._df_prob.sum(axis=1).groupby(level=[level_norm, level_agg]).count()
    
            df_agg_mapping = df_agg.reset_index()[[level_norm_name, level_agg_name]]

            df_norm_var = (
                self._df_prob.groupby(level=level_norm).var()
                .merge(df_agg_mapping, left_index=True, right_on=level_norm_name, how="outer")
                .set_index([level_norm_name, level_agg_name])
            )
        
        df_mult1 = ((df_n_norm - df_n_agg) / df_n_agg / (df_n_norm - 1))

        df_mult = np.sqrt(df_norm_var.mul(df_mult1, axis=0))
        
        self._df_metric = (df_agg - df_norm) / df_mult
        return 
    
    def get_pos_metric(self, threshold = 1.96):
        if self._df_metric is None:
            self.get_metric()
        self._pos_metric = self._df_metric.copy()
        self._pos_metric[self._pos_metric < threshold] = 0

    def get_pos_metric_norm(self):
        self._pos_metric_norm = self._pos_metric.div(self._pos_metric.groupby(level=0).sum())
