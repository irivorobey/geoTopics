import pandas as pd
import numpy as np
import plotly.graph_objs as go

PATH = "data/"

df_country = pd.read_csv(PATH+'countryInfo.csv')
id2name_country=dict(zip(df_country['alpha-2'],df_country['name']))
id2region_country=dict(zip(df_country['alpha-2'],df_country['region']))
id2subregion_country=dict(zip(df_country['alpha-2'],df_country['sub-region']))

id2name_country['XK']='Kosovo'
id2region_country['XK']='Europe'
id2subregion_country['XK']='Southern Europe'

df_topics=pd.read_csv(PATH+'topic_mapping_table_19022024.csv')
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


def plotly_heatmap(df, x_labels, y_labels,
                   title = "Heatmap",
                   x_type = None, y_type=None,
                   x_name="X", y_name="Y", z_name="Z",
                   line_height=20, z_min=None, z_max=None):
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
            colorscale="Viridis",
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
