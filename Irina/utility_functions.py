import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.metrics.pairwise import cosine_distances

PATH = "../data/"

df_country = pd.read_csv(PATH+'Floriana_country_info.csv')
id2name_country=dict(zip(df_country['alpha-2'],df_country['name']))
id2region_country=dict(zip(df_country['alpha-2'],df_country['region']))
id2subregion_country=dict(zip(df_country['alpha-2'],df_country['sub-region']))

id2name_country['XK']='Kosovo'
id2region_country['XK']='Europe'
id2subregion_country['XK']='Southern Europe'

df_topics=pd.read_csv(PATH+'Floriana_topic_mapping.csv')
df_topics['topic_id']=df_topics['topic_id'].apply(lambda x: 'T'+str(x))
id2field_topic=dict(zip(df_topics['subfield_id'],df_topics['field_name']))
id2subfield_topic=dict(zip(df_topics['subfield_id'],df_topics['subfield_name']))

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


def top_n_countries_by_articles(n_countries: int = 20):
    return (
        pd.read_csv(PATH+"df_country_subfield.csv", index_col="country")
        .sum(axis=1)
        .sort_values()
        .tail(n_countries)
        .index
    ).to_list()


def plotly_heatmap(df, x_labels, y_labels,
                   title = "Heatmap",
                   x_type = None, y_type=None,
                   x_name="X", y_name="Y", z_name="Z",
                   line_height=20, z_min=None, z_max=None, colorscale="symmetric"):
    if colorscale == "symmetric":
        colorscale = "RdBu"
    else:
        colorscale = "Blues"

    customdata = np.empty(
        (len(y_labels), len(x_labels), 2),
        dtype=object
    )

    if x_type is None:
        x_full = x_labels
    elif x_type == "country":
        x_full = [f"{id2name_country[c]} ({c})" for c in x_labels]
    elif x_type == "subfield":
        x_full = [f"{id2subfield_topic[int(s)]} ({s})" for s in x_labels]
    else:
        x_full = x_labels

    if y_type is None:
        y_full = y_labels
    elif y_type == "country":
        y_full = [f"{id2name_country[c]} ({c})" for c in y_labels]
    elif y_type == "subfield":
        y_full = [f"{id2subfield_topic[int(s)]} ({s})" for s in y_labels]
    else:
        y_full = y_labels

    customdata[:, :, 0] = np.tile(x_full, (len(y_labels), 1))
    customdata[:, :, 1] = np.tile(np.array(y_full).reshape(-1, 1), (1, len(x_labels)))

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=z_min,
            zmax=z_max,
            customdata=customdata,
            hovertemplate=(
                "<b>"+x_name+": </b>%{customdata[0]}<br>"
                "<b>"+y_name+": </b>%{customdata[1]}<br>"
                "<b>"+z_name+": </b> %{z:.3f}<br>"
                f"<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        height=line_height*len(y_labels),
        # height=40*len(y_labels),
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(autorange="reversed", automargin=True)  # keep top-to-bottom ordering
    )

    return fig


def get_country_stats(df_country_subfield: pd.DataFrame):
    df_country_subfield_norm = get_country_subfield_prob_metric(df_country_subfield)
    return (
        df_country_subfield_norm
        .assign(
            entropy=lambda df: -df.mul(np.log(df.replace(0, np.nan))).sum(axis=1),
            norm_entropy=lambda df: df.entropy / np.log(len(df.columns) - 1),
            total_articles=df_country_subfield.sum(axis=1),
            # log_total_articles=lambda df: np.log(df.total_articles),
            gini=df_country_subfield.apply(gini, axis=1)
        )
        .merge(df_country[["alpha-2", "region"]], left_index=True, right_on="alpha-2", how="left")
        .rename(columns={"alpha-2": "country"})
        .set_index("country")
        .sort_values('total_articles', ascending=False)
        [["norm_entropy", "total_articles", "gini"]]
    )


def get_interest_metric(df_country_subfield: pd.DataFrame,):
    df_country_subfield_norm = (
        df_country_subfield
        .div(df_country_subfield.sum(axis=1), axis=0)
    )

    df_world_subfield_norm = (
        df_country_subfield.sum(axis=0).to_frame().T
        .div(df_country_subfield.sum(axis=0).to_frame().T.sum(axis=1), axis=0)
    )

    df_country_subfield_norm_world_log = np.log(
        df_country_subfield_norm
        .div(df_world_subfield_norm.loc[0])
        .replace(0, np.nan)
    )

    return df_country_subfield_norm_world_log


def get_country_subfield_prob_metric(df_country_subfield: pd.DataFrame,):
    df_country_subfield_norm = (
        df_country_subfield
        .div(df_country_subfield.sum(axis=1), axis=0)
    )
    return df_country_subfield_norm


def get_subfield_stats(df_interest_metric, year):
    return (
        df_interest_metric
        .quantile(0.25)
        .to_frame("q1")
        .assign(
            q2 = df_interest_metric.quantile(0.5),
            q3 = df_interest_metric.quantile(0.75),
            mean_val = df_interest_metric.mean(axis=0),
            std_val = df_interest_metric.std(axis=0),
            year = year
        )
        .reset_index()
    )


def top_n_subfields(row, n=5):
    return [code for code in row.nlargest(n).index.tolist()]


def get_interest_metric_stats(df_interest_metric: pd.DataFrame, df_subfield_stats, year: int):
    df_top_subfields = pd.DataFrame.from_dict(
        df_interest_metric
        .apply(top_n_subfields, axis=1)
        .to_dict(),
        orient="index",
        columns=["top1_subfield", "top2_subfield", "top3_subfield", "top4_subfield", "top5_subfield"]
    )
    return (
        df_interest_metric
        .std(axis=1)
        .to_frame("std_val")
        .assign(
            count_subfield_0_25 = (df_interest_metric.values < df_subfield_stats.q1.values).sum(axis=1),
            count_subfield_25_50 = ((df_interest_metric.values >= df_subfield_stats.q1.values) &
                           (df_interest_metric.values < df_subfield_stats.q2.values)).sum(axis=1),
            count_subfield_50_75 = ((df_interest_metric.values >= df_subfield_stats.q2.values) &
                           (df_interest_metric.values < df_subfield_stats.q3.values)).sum(axis=1),
            count_subfield_75_100 = (df_interest_metric.values >= df_subfield_stats.q3.values).sum(axis=1),
        )
        .merge(df_top_subfields, left_index=True, right_index=True, how="outer")
        .assign(year=year)
        .reset_index()
    )


def get_cosine_distances(df_country_subfield: pd.DataFrame,):
    dist_matrix = cosine_distances(df_country_subfield.values)

    return pd.DataFrame(dist_matrix, index=df_country_subfield.index, columns=df_country_subfield.index)


def get_subfield_tree(df: pd.DataFrame = df_topics, dist_list: list = [1, 0.1, 0.01]):
    domain_id_list = df.domain_id.drop_duplicates().sort_values().to_list()
    domain_list = []
    for domain in domain_id_list:
        field_id_list = df.query(f"domain_id == {domain}").field_id.drop_duplicates().sort_values().to_list()
        field_list = []
        for field in field_id_list:
            subfield_id_list = df.query(
                f"field_id == {field}").subfield_id.drop_duplicates().sort_values().to_list()
            subfield_list = [{"id": subfield, "length": dist_list[2]} for subfield in subfield_id_list]
            field_list.append({
                "id": field,
                "length": dist_list[1],
                "branches": subfield_list,
            })
        domain_list.append({
            "id": domain,
            "length": dist_list[0],
            "branches": field_list,
        })

    subfields_tree = {
        "id": 0,
        "branches": domain_list,
    }

    return subfields_tree


def get_dist_w1_tree(tree, mu_dict, nu_dict):
    subtree = tree.get("branches", None)
    if subtree is None:
        leave_id = tree["id"]
        edge_length = tree.get("length", None)
        mu_id = mu_dict.get(str(leave_id), 0)
        nu_id = nu_dict.get(str(leave_id), 0)
        return mu_id, nu_id, abs(mu_id - nu_id) * edge_length
    else:
        mu_id_array = np.full(len(subtree), 0, dtype=float)
        nu_id_array = np.full(len(subtree), 0, dtype=float)
        dist_sum = 0
        edge_length = tree.get("length", None)
        for i, branch in enumerate(subtree):
            mu_id_array[i], nu_id_array[i], dist_branch = get_dist_w1_tree(branch, mu_dict, nu_dict)
            dist_sum += dist_branch
        if edge_length is None:
            return mu_id_array.sum(), nu_id_array.sum(), dist_sum
        else:
            return mu_id_array.sum(), nu_id_array.sum(), dist_sum + abs(mu_id_array.sum() - nu_id_array.sum()) * edge_length


def get_w1_distances(subfields_tree, df_prob_metric):
    country_list = df_prob_metric.index.to_list()
    country_dist_w1_world = np.full((len(country_list), len(country_list)), np.nan, dtype=float)

    for i, country1 in enumerate(country_list):
        for j, country2 in enumerate(country_list):
            if i == j:
                country_dist_w1_world[i, j] = 0
            if i < j:
                _, _, country_dist_w1_world[i, j] = _, _, country_dist_w1_world[j, i] = get_dist_w1_tree(
                    subfields_tree,
                    mu_dict=df_prob_metric.loc[country1].dropna().to_dict(),
                    nu_dict=df_prob_metric.loc[country2].dropna().to_dict())
    return pd.DataFrame(country_dist_w1_world, index=country_list, columns=country_list)


def count_function(name, df):
    cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
    return (df[cols] == name).sum().sum()


def top_n_cols(row, n=5):
    return [id2subfield_topic.get(int(code), code) for code in row.nlargest(n).index.tolist()]


def get_cluster_dfs(cluster, df_country_subfield_norm_world_norm, df_map,):
    top_cols_df = df_country_subfield_norm_world_norm.apply(lambda r: top_n_cols(r), axis=1)

    df_top_subfields = pd.DataFrame(
        top_cols_df.tolist(),
        index=df_country_subfield_norm_world_norm.index,
        columns=[f"top_{i+1}" for i in range(5)]
    )

    df_map_top = (
        df_map
        .merge(df_top_subfields, left_on="country2", right_index=True)
        .query(f"cluster == {cluster}")
        .sort_values(["top_1", "top_2", "top_3", "top_4", "top_5"])
    )

    df_unique_subfields = (
        pd.DataFrame(df_map_top[["top_1", "top_2", "top_3", "top_4", "top_5"]].values.flatten(), columns=["subfield"])
        .drop_duplicates()  # keep unique
    )

    df_subfields_counted = (
        df_unique_subfields
        .merge(df_topics[["subfield_name", "field_name"]], left_on="subfield", right_on="subfield_name", how="left")
        .drop(["subfield_name"], axis=1)
        .drop_duplicates(subset=["subfield"])  # ensure one row per subfield
        .assign(count_subfields=lambda df: df.subfield.apply(lambda name: count_function(name, df_map_top)))
    )

    df_fields_counted = (
        df_subfields_counted
        .drop("subfield", axis=1)
        .groupby("field_name")
        .sum()
        .div(len(df_map_top.index) * 5)
        .merge(df_topics[["field_name", "domain_name"]].drop_duplicates(), left_index=True, right_on="field_name")
    )
    return df_top_subfields, df_unique_subfields, df_subfields_counted, df_fields_counted