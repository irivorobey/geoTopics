import pandas as pd
import numpy as np

PATH = "data/"

df_country = pd.read_csv(PATH+'countryInfo.csv')
id2name=dict(zip(df_country['alpha-2'],df_country['name']))
id2region=dict(zip(df_country['alpha-2'],df_country['region']))
id2subregion=dict(zip(df_country['alpha-2'],df_country['sub-region']))

id2name['XK']='Kosovo'
id2region['XK']='Europe'
id2subregion['XK']='Southern Europe'

df_topics=pd.read_csv(PATH+'topic_mapping_table_19022024.csv')
df_topics['topic_id']=df_topics['topic_id'].apply(lambda x: 'T'+str(x))
id2name=dict(zip(df_topics['topic_id'],df_topics['topic_name']))
id2field=dict(zip(df_topics['topic_id'],df_topics['field_name']))

def gini(x):
    x = np.asarray(x.dropna(), dtype=float)

    # must be non-negative
    if np.any(x < 0):
        raise ValueError("Gini coefficient is not defined for negative values")

    # all zeros
    if np.allclose(x, 0):
        return 0.0

    # sort
    x = np.sort(x)
    n = x.size

    # Gini formula
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)

def get_subfield_info(subfield_id: int, df_topics: pd.DataFrame = df_topics):
    return (
        df_topics
        [["subfield_id", "subfield_name", "field_name", "domain_name"]]
        .query(f"subfield_id == {subfield_id}")
        .head(1)
    )

def get_country_info(code: str, df_country: pd.DataFrame = df_country):
    return (
        df_country
        [["name", "region", "sub-region", "alpha-2"]]
        .rename(columns={"alpha-2": "code"})
        .query(f"code == \"{code}\"")
    )