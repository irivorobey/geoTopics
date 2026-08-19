import pandas as pd

def get_article_subfield_country(df_year, df_topics, df_institutions):

    df = (
        df_year
        .merge(df_topics[["topic_id", "subfield_id"]],
               left_on="topic", right_on="topic_id", how="left")
        .merge(df_institutions[["institution_id", "country"]],
               left_on="institution", right_on="institution_id", how="left")
        [["id", "subfield_id", "country"]]
        .rename(columns={"id": "article_id"})
        .drop_duplicates()
    )

    # number of countries per article (fully vectorized)
    country_count = (
        df.groupby("article_id")["country"]
        .nunique()
        .rename("n_countries")
    )

    df = df.merge(country_count, on="article_id", how="left")

    return df