import pandas as pd
import numpy as np
import plotly.graph_objs as go
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from matplotlib.patches import Patch
import matplotlib as mpl
import scipy.spatial.distance as spd

from sklearn.metrics.pairwise import cosine_distances

PATH = "../data/"
URL_WORLD = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

df_country = pd.read_csv(PATH+'Floriana_country_info.csv')
id2name_country=dict(zip(df_country['alpha-2'],df_country['name']))
id2region_country=dict(zip(df_country['alpha-2'],df_country['region']))
id2subregion_country=dict(zip(df_country['alpha-2'],df_country['sub-region']))

id2name_country['XK']='Kosovo'
id2region_country['XK']='Europe'
id2subregion_country['XK']='Southern Europe'

id2name_country['Global']='Global'
id2region_country['Global']='Global'
id2subregion_country['Global']='Global'

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


def top_n_countries_by_articles(n_countries: int = 20, year: int = 2023):
    return (
        pd.read_csv(PATH+f"df_country_subfield/{year}.csv", index_col="country")
        .sum(axis=1)
        .sort_values()
        .tail(n_countries)
        .index
    ).to_list()


def escape_latex(s):
    if isinstance(s, str):
        return (
            s.replace('&', '\\&')
             .replace('%', '\\%')
             .replace('$', '\\$')
             .replace('#', '\\#')
             .replace('_', '\\_')
             .replace('{', '\\{')
             .replace('}', '\\}')
             .replace('~', '\\textasciitilde{}')
             .replace('^', '\\^{}')
             .replace('\\', '\\textbackslash{}')
        )
    return s


def update_quantiles(quantiles_thresholds, quantiles_pct=None):
    labels = []

    start_idx = 0
    end_idx = start_idx
    start = quantiles_thresholds[start_idx]
    quantiles = [start]
    quantiles_pct_upd = None
    if quantiles_pct is not None:
        quantiles_pct_upd = [quantiles_pct[start_idx]]

    for i in range(len(quantiles_thresholds)):
        if quantiles_thresholds[i] == start:
            end_idx = i
            continue
        else:
            # process prev
            if start_idx != end_idx:
                quantiles.append(start + 10 ** (-6))
                if quantiles_pct is None:
                    labels.append(f"up to {round(start, 2)}")
                else:
                    quantiles_pct_upd.append(quantiles_pct[end_idx])
                    labels.append(f"Q: {round(quantiles_pct[start_idx], 2)*100} to {round(quantiles_pct[end_idx], 2)*100}%")
                start_idx = end_idx
            quantiles.append(quantiles_thresholds[i])

            if quantiles_pct is None:
                labels.append(f"up to {round(quantiles_thresholds[i], 2)}")
            else:
                quantiles_pct_upd.append(quantiles_pct[i])
                labels.append(f"Q: {round(quantiles_pct[i-1], 2)*100} to {round(quantiles_pct[i], 2)*100}%")
            start = quantiles_thresholds[i]
            start_idx = i
            end_idx = i
    return labels, quantiles, quantiles_pct_upd


def plot_map_continuous(df, column, year="", title="", legend="", country_column="alpha-3", range=None, ax=None, fig=None, animation=False, log=False):
        # ----------------------------
        # Load world geometries
        # ----------------------------

        world = gpd.read_file(URL_WORLD)

        # Merge your data with geometries
        world = world.merge(df, left_on='ADM0_A3', right_on=country_column, how='left').dropna(subset=column)

        # cmap = plt.cm.plasma
        cmap = plt.cm.hsv
        if range is None:
            vmin = world[column].min()
            vmax = world[column].max()
        else:
            vmin = range[0]
            vmax = range[1]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # ----------------------------
        # Create figure with 2 subplots
        # ----------------------------

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        ax.clear()
        # later
        # print(world)
        # --- Move legend to the figure, horizontal ---
        world.plot(column=column, cmap=cmap, norm=norm, edgecolor='black', ax=ax)

        ax.set_title(f"{title} {year}")
        ax.axis("off")

        # if not animation:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for colorbar

        if hasattr(fig, "_cbar"):
            fig._cbar.set_label(legend)
            if log:
                fig._cbar.set_ticks(np.linspace(vmin, vmax, 6))
                fig._cbar.set_ticklabels(np.round(np.exp(np.linspace(vmin, vmax, 6)), 0).astype(int))
            else:
                fig._cbar.set_ticks(np.linspace(vmin, vmax, 6))

        else:
            fig._cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation="vertical",  # or "vertical"
                fraction=0.04,
                pad=0.04
            )
            fig._cbar.set_label(legend)
            if log:
                fig._cbar.set_ticks(np.linspace(vmin, vmax, 6))
                fig._cbar.set_ticklabels(np.round(np.exp(np.linspace(vmin, vmax, 6)), 0).astype(int))
            else:
                fig._cbar.set_ticks(np.linspace(vmin, vmax, 6))


        return fig, ax


def plot_map_distribution(df, column, quantiles_pct, year,
                          map_title, histogram_title, fig=None, ax_map=None, ax_hist=None, hist_range=None):
    quantiles_thresholds = np.quantile(df[column], q=quantiles_pct)

    labels, param, percent = update_quantiles(quantiles_thresholds, quantiles_pct)

    df_map = (
        df
        .assign(
            mapping=lambda df: pd.cut(df[column], bins=param, labels=labels, include_lowest=True)
        )
    )

    # ----------------------------
    # Load world geometries
    # ----------------------------
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)

    # Merge your data with geometries
    world = world.merge(df_map, left_on='ADM0_A3', right_on='alpha-3', how='left')

    label_to_code = {label: i for i, label in enumerate(labels)}
    world['mapping_code'] = world['mapping'].map(label_to_code)
    # ----------------------------
    # Create figure with 2 subplots
    # ----------------------------

    colors = plt.cm.plasma(np.linspace(0, 1, len(quantiles_thresholds)))
    colors_for_labels = colors[-len(labels):]

    cmap_discrete = mpl.colors.ListedColormap(colors_for_labels)

    if fig is None:
        fig, (ax_map, ax_hist) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(12,6),
            gridspec_kw={"height_ratios":[3,1], "hspace":0.2}
        )
    ax_hist.clear()
    ax_map.clear()

    if fig.legends:   # fig.legends is a list of figure legends
        for lg in fig.legends:
            lg.remove()
    # --- 1️⃣ Plot choropleth map ---
    g = world.plot(
        column='mapping',       # threshold group
        cmap='Set3',            # discrete colormap
        legend=True,            # temporarily create legend
        ax=ax_map,
        edgecolor='black'
    )
    # fig.set_title(f"Country losses ({year})")
    ax_map.axis('off')

    # --- Move legend to the figure, horizontal ---
    world.plot(column='mapping_code', cmap=cmap_discrete, edgecolor='black', ax=ax_map)

    # 3️⃣ Create legend handles manually
    legend_handles = [Patch(facecolor=color, edgecolor='black', label=label)
                      for label, color in zip(labels, colors_for_labels)]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(labels),
               title=f"{map_title}: {year}", frameon=False, bbox_to_anchor=(0.5, 0.95), title_fontsize=12)

    ax_map.get_legend().remove()  # remove old legend

    # --- 2️⃣ Plot histogram ---
    # values = df[column].values
    values = df[column].dropna().values

    bins_hist = 50
    counts, bin_edges, _ = ax_hist.hist(values, bins=bins_hist, range=hist_range, color='black', edgecolor='white')
    max_height = counts.max()

    # Threshold background colors
    colors = colors_for_labels
    bins_threshold = list(zip(param[:-1], param[1:]))

    for (lo, hi), color in zip(bins_threshold, colors):
        ax_hist.axvspan(lo, hi, ymin=0, ymax=1, facecolor=color, alpha=0.3)

    # Vertical lines and labels
    for i, (lo, hi) in enumerate(bins_threshold):
        ax_hist.axvline(lo, color='black', linestyle='--', linewidth=1)
        ax_hist.axvline(hi, color='black', linestyle='--', linewidth=1)
        ax_hist.text(hi, max_height*0.85, f"{hi:.2f}", ha='right', va='bottom', fontsize=10)

    # Labels and title
    ax_hist.set_xlabel("Loss share")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{histogram_title} ({year})")

    fig.subplots_adjust(
        top=0.8,  # leave space for title or legend above
        bottom=0.1,  # space below
        hspace=0.25  # space between subplots
    )

    # plt.tight_layout()

    return df_map, ax_map, ax_hist


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


def get_global_full(df_year, df_topics_maxime, df_institutions):
    # print("df_global started")
    df_global = (
        df_year
        .merge(df_topics_maxime[["topic_id", "subfield_id"]], left_on="topic", right_on="topic_id", how="left")
        .merge(df_institutions, left_on="institution", right_on="institution_id", how="left")
        [["year", "id", "authors", "institution_id", "country", "subfield_id"]]
        .rename(columns={"id": "article_id", "authors": "author_id"})
        .groupby(["year", "article_id", "subfield_id", "country"], as_index=False)
        .agg(list)
        .drop_duplicates(subset=["article_id", "subfield_id", "country"])
    )
    # print("df_global completed")
    df_article_authors = (
        df_global
        .groupby("article_id", as_index=False)
        .author_id
        .sum()
        .rename(columns={"author_id": "all_author_id"})
        .assign(all_authors_count=lambda df: [len(a) for a in df.all_author_id])
    )
    df_article_countries = (
        df_global
        .groupby("article_id", as_index=False)
        .country
        .agg(list)
        .rename(columns={"country": "all_countries"})
        .assign(all_countries_count=lambda df: [len(a) for a in df.all_countries])
    )
    # print("df_global_full started")
    df_global_full = (
        df_global
        .assign(
            authors_count=lambda df: [len(a) for a in df.author_id],
        )
        .merge(df_article_authors, on="article_id", how="left")
        .assign(authors_share=lambda df: df.authors_count / df.all_authors_count)
        # .query("authors_share != 1")
        .merge(df_article_countries, on="article_id", how="left")
        .assign(countries_share=lambda df: 1 / df.all_countries_count)
        # .query("countries_share != 1")
    )
    return df_global_full


def get_global_comparison(df_global_full, year):
    return (
        df_global_full
        .groupby("subfield_id")
        .article_id.nunique().to_frame("counts_all")
        .assign(
            counts_individual = df_global_full.query("countries_share == 1").groupby("subfield_id").article_id.nunique(),
            counts_collab = df_global_full.query("countries_share != 1").groupby("subfield_id").article_id.nunique(),
            # check_sum=lambda df:df.counts_individual + df.counts_collab,
            # check_x_countries = df_global_full.groupby("subfield_id").countries_share.sum(),
            # check_x_authors = df_global_full.groupby("subfield_id").authors_share.sum(),
            probability_all = lambda df: df.counts_all / df.counts_all.sum(),
            probability_individual = lambda df: df.counts_individual / df.counts_individual.sum(),
            probability_collab = lambda df: df.counts_collab / df.counts_collab.sum(),
            counts_individual_pct = lambda df: df.counts_individual / df.counts_all * 100,
            counts_collab_pct = lambda df: df.counts_collab / df.counts_all * 100,
            year=year,
        )
        .reset_index("subfield_id")
    )


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

def get_counts_year(year, df_global, path="df_country_subfield/"):
    df_country_subfield_indiv = pd.concat([
            pd.read_csv(PATH+path+f"individual_{year}.csv", index_col=0),
            df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_individual.to_frame("Global").T
        ], axis=0)

    df_country_subfield_collab = pd.concat([
        pd.read_csv(PATH+path+f"collaborative_{year}.csv", index_col=0),
        df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_collab.to_frame("Global").T
    ], axis=0)

    df_country_subfield_all = pd.concat([
            pd.read_csv(PATH+path+f"{year}.csv", index_col=0),
            df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_all.to_frame("Global").T
        ], axis=0)

    df_country_shares = (
        df_country_subfield_indiv
        .sum(axis=1)
        .to_frame("counts_indiv")
        .assign(counts_collab=df_country_subfield_collab.sum(axis=1))
        .merge(df_country[["alpha-2", "alpha-3", "name", "region", "sub-region"]], left_index=True, right_on="alpha-2")
    )
    df_country_shares[["counts_indiv", "counts_collab"]] = (
        df_country_shares[["counts_indiv", "counts_collab"]].fillna(0)
    )
    return df_country_shares

def get_losses_year(year, df_global, path="df_country_subfield/"):
    df_country_subfield_indiv = pd.concat([
            pd.read_csv(PATH+path+f"individual_{year}.csv", index_col=0),
            df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_individual.to_frame("Global").T
        ], axis=0)

    df_country_subfield_collab = pd.concat([
        pd.read_csv(PATH+path+f"collaborative_{year}.csv", index_col=0),
        df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_collab.to_frame("Global").T
    ], axis=0)

    df_country_subfield_all = pd.concat([
            pd.read_csv(PATH+path+f"{year}.csv", index_col=0),
            df_global.query("year == @year").assign(subfield=lambda df: df.subfield_id.astype(str)).set_index("subfield").counts_all.to_frame("Global").T
        ], axis=0)
    df_country_shares = (
        df_country_subfield_indiv
        .sum(axis=1)
        .div(df_country_subfield_all.sum(axis=1) / 100)
        .to_frame("counts_indiv_pct")
        .assign(counts_collab_pct=df_country_subfield_collab.sum(axis=1).div(df_country_subfield_all.sum(axis=1) / 100))
        .merge(df_country[["alpha-2", "alpha-3", "name", "region", "sub-region"]], left_index=True, right_on="alpha-2")
    )
    df_country_shares[["counts_indiv_pct", "counts_collab_pct"]] = (
        df_country_shares[["counts_indiv_pct", "counts_collab_pct"]].fillna(0)
    )
    return df_country_shares


def get_subfield_stats(df_country_subfield, df_interest_metric, year):

    return (
        df_interest_metric
        .quantile(0.25)
        .to_frame("q1")
        .assign(
            q2 = df_interest_metric.quantile(0.5),
            q3 = df_interest_metric.quantile(0.75),
            mean_val = df_interest_metric.mean(axis=0),
            std_val = df_interest_metric.std(axis=0),
            counts_countries = df_interest_metric.count(axis=0),
            counts_articles = df_country_subfield.sum(axis=0),
            year = year
        )
        .reset_index()
    )


def top_n_columns(row, n=5):
    if n == 1:
        return [code for code in row.nlargest(n).index.tolist()][0]
    else:
        return [code for code in row.nlargest(n).index.tolist()]


def bottom_n_columns(row, n=5):
    if n == 1:
        return [code for code in row.nsmallest(n).index.tolist()][0]
    else:
        return [code for code in row.nsmallest(n).index.tolist()]


def get_interest_metric_stats(df_interest_metric: pd.DataFrame, df_subfield_stats, year: int):
    df_top_subfields = pd.DataFrame.from_dict(
        df_interest_metric
        .apply(top_n_columns, axis=1)
        .to_dict(),
        orient="index",
        columns=["top1_subfield", "top2_subfield", "top3_subfield", "top4_subfield", "top5_subfield"]
    )
    interest_metric_array = df_interest_metric.values
    return (
        df_interest_metric
        .std(axis=1)
        .to_frame("std_val")
        .assign(
            count_subfield_0_25 = (interest_metric_array < df_subfield_stats.q1.values).sum(axis=1),
            count_subfield_25_50 = ((interest_metric_array >= df_subfield_stats.q1.values) &
                           (interest_metric_array < df_subfield_stats.q2.values)).sum(axis=1),
            count_subfield_50_75 = ((interest_metric_array >= df_subfield_stats.q2.values) &
                           (interest_metric_array < df_subfield_stats.q3.values)).sum(axis=1),
            count_subfield_75_100 = (interest_metric_array >= df_subfield_stats.q3.values).sum(axis=1),
        )
        .merge(df_top_subfields, left_index=True, right_index=True, how="outer")
        .assign(year=year)
        .reset_index()
    )


def get_cosine_distances(df_country_subfield: pd.DataFrame,):
    dist_matrix = cosine_distances(df_country_subfield.values)

    return pd.DataFrame(dist_matrix, index=df_country_subfield.index, columns=df_country_subfield.index)

def get_field_tree(df: pd.DataFrame = df_topics, dist_list: list = [1, 0.1]):
    domain_id_list = df.domain_id.drop_duplicates().sort_values().to_list()
    domain_list = []
    for domain in domain_id_list:
        field_id_list = df.query(f"domain_id == {domain}").field_id.drop_duplicates().sort_values().to_list()
        field_list = [{"id": str(field), "length": dist_list[1]} for field in field_id_list]
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


def get_w1_distances(subfields_tree, df_prob_metric, subfield_mode = True):
    if subfield_mode:
        subfield1 = "2713"
        subfield2 = "1904"
        _, _, w1_max = get_dist_w1_tree(subfields_tree, {subfield1: 1}, {subfield2: 1})
    else:
        subfield1 = "13"
        subfield2 = "22"
        _, _, w1_max = get_dist_w1_tree(subfields_tree, {subfield1: 1}, {subfield2: 1})

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
    return pd.DataFrame(country_dist_w1_world, index=country_list, columns=country_list).div(w1_max)


def get_distance_stats(df_distance, year):
    df_distance_nan_diag = df_distance.replace(0, np.nan)
    distance_array = df_distance_nan_diag.values

    return (
        df_distance_nan_diag
        .mean(axis=1)
        .to_frame("mean_values")
        .assign(
            std_values=np.nanstd(distance_array, axis=1),
            q1=np.nanpercentile(distance_array, 25, axis=1),
            q2=np.nanmedian(distance_array, axis=1),
            q3=np.nanpercentile(distance_array, 75, axis=1),
            count_0_1=(distance_array <= 1).sum(axis=1),
            count_1_2=(distance_array > 1).sum(axis=1),
            share_0_1=(distance_array <= 1).sum(axis=1) / df_distance_nan_diag.count(axis=1),
            share_1_2=(distance_array > 1).sum(axis=1) / df_distance_nan_diag.count(axis=1),
            furtherest=df_distance_nan_diag.apply(lambda row: top_n_columns(row, 1), axis=1),
            closest=df_distance_nan_diag.apply(lambda row: bottom_n_columns(row, 1), axis=1),
            furtherest_dist=df_distance_nan_diag.max(axis=1),
            closest_dist=df_distance_nan_diag.min(axis=1),
            range_dist=lambda df: df.furtherest_dist - df.closest_dist,
            year=year
        )
        .reset_index()
    )


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

    df_individual_subfields = (
        pd.DataFrame(df_map_top[["top_1", "top_2", "top_3", "top_4", "top_5"]].values.flatten(), columns=["subfield"])
        .drop_duplicates()  # keep individual
    )

    df_subfields_counted = (
        df_individual_subfields
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
    return df_top_subfields, df_individual_subfields, df_subfields_counted, df_fields_counted


def get_jensen_shannon_distances(df_metric):
    n_countries = len(df_metric.index)
    dist_array = np.full((n_countries, n_countries), np.nan)
    for i in range(n_countries):
        for j in range(n_countries):
            dist_array[i, j] = spd.jensenshannon(df_metric.iloc[i].fillna(0), df_metric.iloc[j].fillna(0))
    return pd.DataFrame(dist_array, index=df_metric.index, columns=df_metric.index)


def get_jaccard_distances(df_metric):
    n_countries = len(df_metric.index)
    dist_array = np.full((n_countries, n_countries), np.nan)
    for i in range(n_countries):
        for j in range(n_countries):
            dist_array[i, j] = spd.jaccard(df_metric.iloc[i].fillna(0) > 0, df_metric.iloc[j].fillna(0) > 0)
    return pd.DataFrame(dist_array, index=df_metric.index, columns=df_metric.index)


def get_jaccard(df, df1, df2, col_name="jaccard", col_df2=None):
    for c in df.index:
        if col_df2 is None:
            if (c in df1.index) and (c in df2.index):
                df.loc[c, col_name] = spd.jaccard(df1.loc[c].fillna(0) > 0, df2.loc[c].fillna(0) > 0)
            else:
                df.loc[c, col_name] = 1
        else:
            if c in df1.index:
                df.loc[c, col_name] = spd.jaccard(df1.loc[c] > 0, df2[col_df2] > 0)
            else:
                df.loc[c, col_name] = 1

def get_w1_tree(df, subfield_tree, df1, df2=None, col_name="w1_tree", dict2=None, w1_max=None):
    if w1_max is None:
        subfield1 = "2713"
        subfield2 = "1904"
        _, _, w1_max = get_dist_w1_tree(subfield_tree, {subfield1: 1}, {subfield2: 1})
    for c in df.index:
        if dict2 is None:
            if (c in df1.index) and (c in df2.index):
                _, _, tmp = get_dist_w1_tree(subfield_tree,
                                                          df1.loc[c].dropna().to_dict(),
                                                          df2.loc[c].dropna().to_dict())
                df.loc[c, col_name] = tmp / w1_max
            else:
                df.loc[c, col_name] = 1
        else:
            if c in df1.index:
                _, _, tmp = get_dist_w1_tree(subfield_tree,
                                                             df1.loc[c].dropna().to_dict(),
                                                             dict2)
                df.loc[c, col_name] = tmp / w1_max
            else:
                df.loc[c, col_name] = 1