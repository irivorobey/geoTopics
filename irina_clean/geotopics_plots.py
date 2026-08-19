import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


from collections import defaultdict
import numpy as np
import pandas as pd

import subfield_tree as sft
# import w1_tree as wt

PATH = "../data_clean/"

df_subfield = (
    pd.read_csv(PATH+"df_topics.csv")
    [["subfield_id", "subfield_name", "field_id", "field_name", "domain_id", "domain_name"]]
    .drop_duplicates()
    .sort_values(["domain_id", "field_id", "subfield_id"])
)
df_subfield.subfield_id = df_subfield.subfield_id.astype(int)
subfield_tree = sft.SubfieldTree(df_subfield)

def white_to_color(color, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"white_to_{color}",
        [(0.0, "white"), (1, color)],
        N=n
    )

DOMAIN_COLORS = {'Life Sciences':'#81b29a', 'Physical Sciences':'#3d405b', 'Social Sciences':'#e07a5f', 'Health Sciences':'#f4f1de'}

FIELD_COLORS = {}

for domain_name in DOMAIN_COLORS.keys():
    field_list = (
        subfield_tree.df_subfield
        [["field_name", "domain_name"]]
        .drop_duplicates()
        .query("domain_name == @domain_name")
    ).field_name.to_list()

    cmap = white_to_color(DOMAIN_COLORS[domain_name])
    FIELD_COLORS = FIELD_COLORS | dict(zip(field_list, [mcolors.to_hex(c, keep_alpha=True) for c in cmap(np.linspace(0.2, 1, len(field_list)+1))][1:]))

FIELD_NAME_DICT = {
    # "Environmental Science": "Environ. Sc.",
    # "Physics and Astronomy": "Phys. & Astron.",
    "Earth and Planetary Sciences": "Earth & Planet.",
    "Economics, Econometrics and Finance": "Econ. & Finance",
    "Pharmacology, Toxicology and Pharmaceutics": "Pharma. & Toxicology",
    "Veterinary": "Veterinary",
    "Business, Management and Accounting": "Business & Account.",
    "Immunology and Microbiology": "Immun. & Microbio.",
    "Biochemistry, Genetics and Molecular Biology": "Biochem. & Genetics",
    "Arts and Humanities": "Arts & Humanities",
    "Agricultural and Biological Sciences": "Agricult. & Bio. Sc.",
    "Dentistry": "Dentistry",
    "Nursing": "Nursing"
}


def get_domain_pct(df, subfield_tree, domain):
    return int((
        df.to_frame("profile")
        .merge(subfield_tree.df_subfield[["subfield_id", "domain_id"]], left_index=True, right_on="subfield_id")
        .drop(columns="subfield_id")
        .query("domain_id == @domain")
        .profile
        .sum()
    ) * 100)


def plot_two_profiles(df1, df2, subfield_tree, w1_tree, labels=["Profile 1", "Profile 2"], title="Tree-Wasserstein distance", names=None):
    fig, ax = plt.subplots(figsize = (20, 5))

    ax.bar(df1.index.astype(str), df1.values, color="#1F6173", alpha=0.3)
    ax.scatter(df1.index.astype(str), df1.values, s=10, color="#1F6173", label=labels[0])

    ax.bar(df2.index.astype(str), df2.values, color="#D8370A", alpha=0.3)
    ax.scatter(df2.index.astype(str), df2.values, s=10, color="#D8370A", label=labels[1])

    x_ticks = []
    x_ticks_labels = []

    for i, domain in enumerate(subfield_tree.domain_list):
        start_domain = subfield_tree.df_subfield.query("domain_id == @domain").subfield_id.iloc[0].astype(str)
        domain_pct1 = get_domain_pct(df1, subfield_tree, domain)
        domain_pct2 = get_domain_pct(df2, subfield_tree, domain)
        ax.axvline(start_domain, color="grey")
        x_ticks.append(start_domain)
        x_ticks_labels.append(subfield_tree.domain_name_list[i]+f":\n{labels[0]} = {domain_pct1}%,\n{labels[1]} = {domain_pct2}%")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontdict={"horizontalalignment": "left"}, fontsize=14)

    for field in subfield_tree.field_list:
        start_domain = subfield_tree.df_subfield.query("field_id == @field").subfield_id.iloc[0].astype(str)
        ax.axvline(start_domain, color="grey", linestyle="--", alpha=0.2)

    plt.legend(fontsize=14)

    ax.grid(axis="y", alpha=0.3)
    
    wt_dist = np.round(w1_tree.dist(df1.to_frame(labels[0]).T, df2.to_frame(labels[1]).T).iloc[0, 0], 2)

    if names is None:
        names = labels

    _ = ax.set_title(f"{names[0]} vs {names[1]}: W-tree = {wt_dist}", fontsize=16)
    _ = ax.set_ylabel("Subfield probability", fontsize=14)


def plot_profile_pie_fields(df_profile, subfield_tree, plot_list=None, fig=None, gs=None, gs_idx=0, ax=None, title="Profiles"):
    # print(country_code)
    df = (
        df_profile
        .groupby(subfield_tree.field_name_dict, axis=1)
        .sum()
        .T
        .loc[subfield_tree.field_name_list]
    )

    if plot_list is None:
        plot_list = df.columns.tolist()

    if gs is None:
        fig = plt.figure(figsize=(5 * len(plot_list), 5))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        subspec = gs[0]
    else:
        if fig is None:
            fig = plt.gcf()  # use current figure if not explicitly provided
        subspec = gs[gs_idx, :]

    top_gs = subspec.subgridspec(1, len(plot_list), wspace=0)
    axes_pies = [fig.add_subplot(top_gs[0, i]) for i in range(len(plot_list))]
    if ax is None:
        plt.suptitle(f"{title}", fontsize=18)
    else:
        ax.set_title(title, fontsize=16, fontweight="bold", va="top", ha="center")
    
    for ax, col in zip(axes_pies, plot_list):

        values = df[col].loc[subfield_tree.field_name_list].values
        labels = subfield_tree.field_name_list
        colors = [FIELD_COLORS.get(f) for f in labels]

        ax.pie(
            values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1),
            autopct=lambda p: f"{p:.1f}%" if p >= 5 else None,
            textprops={"fontsize": 14},
            pctdistance=0.8
        )
        ax.set_xlabel(str(col), fontsize=16)
        # ax.set_title(str(col), fontsize=16)
        ax.set_aspect("equal")


def plot_field_legend(gs, subfield_tree, field_colors, field_name_dict):

    ax = plt.subplot(gs[-1, 0])
    ax.axis("off")

    field_order = subfield_tree.field_name_list

    phys = [f for f in subfield_tree.field_name_list if subfield_tree.field_domain_name_dict.get(f) == "Physical Sciences"]
    mid = len(phys) // 2
    phys_1, phys_2 = phys[:mid], phys[mid:]

    field_to_column = {}

    for f in field_order:
        d = subfield_tree.field_domain_name_dict.get(f)

        if d == "Health Sciences":
            field_to_column[f] = "Health Sciences"
        elif d == "Life Sciences":
            field_to_column[f] = "Life Sciences"
        elif d == "Social Sciences":
            field_to_column[f] = "Social Sciences"
        elif f in phys_1:
            field_to_column[f] = "Physical Sciences (1)"
        else:
            field_to_column[f] = "Physical Sciences (2)"

    groups = defaultdict(list)
    for f in subfield_tree.field_name_list:
        groups[field_to_column[f]].append(f)

    panel_order = [
        "Health Sciences",
        "Physical Sciences (1)",
        "Physical Sciences (2)",
        "Life Sciences",
        "Social Sciences"
    ]

    x_positions = [0.02, 0.22, 0.42, 0.62, 0.82]

    for x, panel in zip(x_positions, panel_order):

        ax.text(x, 0.95, panel, fontsize=14, weight="bold",
                transform=ax.transAxes)

        y = 0.85
        y_step = 0.08

        for f in groups[panel]:

            ax.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.02),
                    0.02, 0.03,
                    color=field_colors[f],
                    transform=ax.transAxes
                )
            )

            label = field_name_dict.get(f, f)

            ax.text(
                x + 0.03, y,
                label,
                transform=ax.transAxes,
                va="center",
                fontsize=14
            )

            y -= y_step


def plot_panel_profiles_pie_fields(df, subfield_tree, n_rows, idx_rows, level_rows=0, title_rows=None, plot_list=None):
    fig = plt.figure(figsize=(18, 6 * (n_rows + 1)))

    gs = gridspec.GridSpec(
        n_rows + 1, 1,   # one column, many rows
        height_ratios=[1]*n_rows + [0.8],
        hspace=0.3,
        # vspace=0.1
    )

    if title_rows is None:
        title_rows = idx_rows

    for i, (idx_rows, title_rows) in enumerate(zip(idx_rows, title_rows)):
        ax = plt.subplot(gs[i, :])
        plot_profile_pie_fields(
            df.xs(idx_rows, level=level_rows), subfield_tree, plot_list=plot_list,
            gs=gs, gs_idx=i, title=title_rows, ax=ax
        )
        ax.axis("off")

    _ = plot_field_legend(gs, subfield_tree, FIELD_COLORS, FIELD_NAME_DICT)

    plt.tight_layout()


def draw_polar(ax, df, year, r_max, color_dict_tmp):
    position_list = (
        df_subfield
        .reset_index(drop=True)
        .reset_index(names=["position"])
        .groupby("domain_id", as_index=False).first()
        .sort_values(["domain_id", "field_id"], ascending=[True, True])
        .position.to_list()
    )

    position_field_list = (
        df_subfield
        .reset_index(drop=True)
        .reset_index(names=["position"])
        .groupby("field_id", as_index=False).first()
        .sort_values(["domain_id", "field_id"], ascending=[True, True])
        .position.to_list()
    )

    names_field_list = (
        df_subfield
        .reset_index(drop=True)
        .reset_index(names=["position"])
        .groupby("field_id", as_index=False).first()
        .sort_values(["domain_id", "field_id"], ascending=[True, True])
        .field_name.to_list()
    )

    field_dict = {
        "Environmental Science": "Environ. Sc.",
        "Physics and Astronomy": "Phys. & Astron.",
        "Earth and Planetary Sciences": "Earth & Planet.",
        "Economics, Econometrics and Finance": "Econ. & Finance         ",
        "Pharmacology, Toxicology and Pharmaceutics": "        Pharma. & Toxicology",
        "Veterinary": "                  Veterinary",
        "Business, Management and Accounting": "Business & Account.",
        "Immunology and Microbiology": "Immun. & Microbio.",
        "Biochemistry, Genetics and Molecular Biology": "Biochem. & Genetics",
        "Arts and Humanities": "Arts & Humanities",
        "Agricultural and Biological Sciences": "Agricult. & Bio. Sc.",
        "Dentistry": "   Dentistry",
        "Nursing": "  Nursing"
    }

    r_min = r_max * 2
    coef = 3

    n = df.shape[1]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)

    for i, row in df.iterrows():
        r = row.values

        theta_closed = np.append(theta, theta[0])
        r_closed = np.append(r, r[0])

        ax.plot(theta_closed, r_closed, color=color_dict_tmp[i], alpha=1, linewidth=2.5)

        ax.fill_between(theta_closed, 0, np.clip(r_closed, 0, None),
                        alpha=0.3, color=color_dict_tmp[i])

        ax.fill_between(theta_closed, 0, np.clip(r_closed, None, 0),
                        alpha=0.3, color=color_dict_tmp[i])

    ax.set_ylim(-r_min, coef * r_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    # ax.set_title(str(year), fontsize=20)


    # example angles
    n = 252
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)

    # --- grey zones (example: 3 segments) ---
    zones = [
        (0, np.pi/3),
        (np.pi, 4*np.pi/3)
    ]
    zones = [
        (position_list[0] * 2*np.pi/n, position_list[1] * 2*np.pi/n),
        (position_list[1] * 2*np.pi/n, position_list[2] * 2*np.pi/n),
        (position_list[2] * 2*np.pi/n, position_list[3] * 2*np.pi/n),
        (position_list[3] * 2*np.pi/n, 2*np.pi)
    ]      

    zones_colors = ['#f1efe1', '#414357', '#d1836f', '#86ac9a']

    theta = np.linspace(0, 2*np.pi, df.shape[1], endpoint=False)


    for start, end in zones:
        ax.bar(
            x=(start + end) / 2,      # center angle
            height=0.2*r_max,
            width=(end - start),
            # top=20,
            bottom=r_max,
            color=zones_colors[zones.index((start, end))],
            alpha=1,
            edgecolor=None,
            # zorder=0
        )
        ax.plot([start, start], [-r_min, r_max+2], color="black", alpha=1, linewidth=1, zorder=0)
        # ax.plot([end, end], [-r_min, r_max+2], color="black", alpha=1, linewidth=1, zorder=0)

    theta = np.linspace(0, 2*np.pi, 500)

    ax.plot(
        theta,
        np.zeros_like(theta),
        color='black',
        linewidth=1,
        zorder=2
    )

    ax.plot(
        theta,
        r_max* np.ones_like(theta),
        color='black',
        linewidth=1,
        zorder=2
    )

    ax.grid(False)

    for i, pos in enumerate(position_field_list):
        a = pos * 2*np.pi/n
        if names_field_list[i] == "Veterinary":
            ax.plot([a, a], [-r_min, 3.5 * r_max],
                    color="black", alpha=1, linewidth=1, zorder=0)
        else:
            ax.plot([a, a], [-r_min, coef * r_max],
                    color="black", alpha=1, linewidth=1, zorder=0)
        # print(i, a / np.pi * 180)
        if (a / np.pi * 180) <= 90 or (a / np.pi * 180) >= 270:
            ha = 'left'
            va = "bottom"
            angle = a + 0.01
            rotation = angle / np.pi * 180  
        else:
            ha = 'right'
            va = "top"
            angle = a + 0.01
            rotation = (a + 0.025) / np.pi * 180 + 180
        if names_field_list[i] in field_dict.keys():
            test_text = field_dict[names_field_list[i]].split(" ")
        else:
            test_text = names_field_list[i].split(" ")
        if (len(test_text) > 3) and  not (names_field_list[i] in field_dict.keys()):
            test_text = " ".join(test_text[:2]) + "\n" + " ".join(test_text[2:])
        else:
            test_text = " ".join(test_text)
        ax.text(angle, r_max+0.3*r_max, test_text, ha=ha, va=va, rotation=rotation, rotation_mode='anchor', fontsize=20)

    ax.set_ylim(-r_min, coef * r_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)

    # plt.savefig(f"spec_art_{year}_no_fields.svg",format="svg",
    #     transparent=True,)
    _ = ax.text(np.pi / 2, 5*r_max, year, va="center", ha="center", fontsize=24, fontweight="bold")


def dyn_clusters_preprocess(
    df_mapping,
    year_list,
    color_dict_tmp,
    n_countries_total=195,
    width=0.3,
):
    """
    Preprocess all information needed for the dynamic-cluster plot.

    This function does NOT depend on country_list, so it only needs
    to be run once if you want to change the highlighted countries.

    Returns
    -------
    dict
        All precomputed information needed for plotting.
    """

    # ------------------------------------------------------------
    # 1. Cluster sizes / rectangles
    # ------------------------------------------------------------

    df_tmp = (
        df_mapping
        .query("year in @year_list")
        .groupby(["dynamic_cluster", "year"], as_index=False)
        .country.nunique()
        .sort_values(
            ["year", "dynamic_cluster"],
            ascending=[False, True]
        )
        .assign(
            country_share=lambda df: df.country / n_countries_total
        )
        .reset_index(drop=True)
    )

    total_share = (
        df_tmp
        .query("year == 2023")
        .country_share
        .sum()
    )

    # One row in df_tmp -> one rectangle
    rectangles = []
    rectangle_colors = []

    for i, year_loop in enumerate(year_list):

        df_year = df_tmp.query("year == @year_loop")

        vert_coord = 0

        for _, row in df_year.iterrows():

            cluster = row["dynamic_cluster"]
            share = row["country_share"]

            x = len(year_list) - i - width
            y = vert_coord
            w = width
            h = share

            rectangles.append((x, y, w, h))
            rectangle_colors.append(color_dict_tmp[cluster])

            vert_coord += share

    # ------------------------------------------------------------
    # 2. Pre-split mapping by year
    # ------------------------------------------------------------

    year_df = {
        year: (
            df_mapping
            .query("year == @year")
            [["country", "dynamic_cluster"]]
            .copy()
        )
        for year in year_list
    }

    # ------------------------------------------------------------
    # 3. Precompute all country flows
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 3. Precompute all country flows
    # ------------------------------------------------------------

    country_flows = [None] * len(rectangles)
    cluster_flows = [None] * len(rectangles)

    prev_y1 = {}

    for k, year_loop in enumerate(year_list[1:]):

        year_next = year_list[k]

        if k < len(year_list) - 3:
            year_prev = year_list[k + 2]
            year_prev_prev = year_list[k + 3]

        elif k < len(year_list) - 2:
            year_prev = year_list[k + 2]
            year_prev_prev = None

        else:
            year_prev = None
            year_prev_prev = None

        # --------------------------------------------------------
        # Build transition dataframe with explicit column names
        # --------------------------------------------------------

        df_transition = (
            year_df[year_loop]
            .rename(columns={
                "dynamic_cluster": "dynamic_cluster_cur"
            })
            .merge(
                year_df[year_next].rename(columns={
                    "dynamic_cluster": "dynamic_cluster_next"
                }),
                on="country",
                how="left",
            )
        )

        if year_prev is not None:
            df_transition = df_transition.merge(
                year_df[year_prev].rename(columns={
                    "dynamic_cluster": "dynamic_cluster_prev"
                }),
                on="country",
                how="left",
            )

        if year_prev_prev is not None:
            df_transition = df_transition.merge(
                year_df[year_prev_prev].rename(columns={
                    "dynamic_cluster": "dynamic_cluster_prev_prev"
                }),
                on="country",
                how="left",
            )

        # --------------------------------------------------------
        # Process every current cluster
        # --------------------------------------------------------

        clusters_this_year = (
            df_tmp
            .query("year == @year_loop")
            ["dynamic_cluster"]
            .tolist()
        )

        # IMPORTANT:
        # This must be shared across all current clusters
        # for this year transition.

        next_year_dict = {}

        # Initialize destination-cluster positions ONCE

        all_next_clusters = (
            df_transition["dynamic_cluster_next"]
            .dropna()
            .unique()
        )

        for next_cluster in all_next_clusters:
            next_cluster_idx = df_tmp.index[
                (df_tmp["year"] == year_next)
                & (
                    df_tmp["dynamic_cluster"]
                    == next_cluster
                )
            ][0]

            next_year_dict[next_cluster] = (
                rectangles[next_cluster_idx][1]
            )

        for cluster_loop in clusters_this_year:

            df_cluster = df_transition.query(
                "dynamic_cluster_cur == @cluster_loop"
            ).copy()

            # Same sorting as in the original code
            if year_prev_prev is not None:

                df_cluster = df_cluster.sort_values(
                    [
                        "dynamic_cluster_prev_prev",
                        "dynamic_cluster_prev",
                        "dynamic_cluster_next",
                    ]
                )

            elif year_prev is not None:

                df_cluster = df_cluster.sort_values(
                    [
                        "dynamic_cluster_prev",
                        "dynamic_cluster_next",
                    ]
                )

            else:

                df_cluster = df_cluster.sort_values(
                    ["dynamic_cluster_next"]
                )

            n_countries = len(df_cluster)

            # ----------------------------------------------------
            # Current rectangle
            # ----------------------------------------------------

            cur_cluster_idx = df_tmp.index[
                (df_tmp["year"] == year_loop)
                & (
                    df_tmp["dynamic_cluster"]
                    == cluster_loop
                )
            ][0]

            country_step = (
                rectangles[cur_cluster_idx][3]
                / n_countries
            )

            # ----------------------------------------------------
            # Country coordinates
            # ----------------------------------------------------

            y_cur = rectangles[cur_cluster_idx][1]

            country_flow_list = []
            cluster_flow_list = []

            for _, row in df_cluster.iterrows():

                country = row["country"]
                next_cluster = row["dynamic_cluster_next"]

                y1 = y_cur

                # Country disappears in the next year
                if pd.isna(next_cluster):
                    y_cur += country_step
                    continue

                y2 = prev_y1.get(
                    country,
                    next_year_dict[next_cluster],
                )

                x1 = len(year_list) - k - 1
                x2 = len(year_list) - k - width

                country_flow_list.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        country,
                    )
                )

                # Short horizontal segment used for labels
                x1_cluster = (
                    len(year_list) - k - width
                )

                x2_cluster = (
                    len(year_list) - k
                )

                cluster_flow_list.append(
                    (
                        x1_cluster,
                        y2,
                        x2_cluster,
                        y2,
                        country,
                    )
                )

                # Update positions
                prev_y1[country] = y_cur

                next_year_dict[next_cluster] += country_step

                y_cur += country_step

            country_flows[cur_cluster_idx] = (
                country_flow_list
            )

            cluster_flows[cur_cluster_idx] = (
                cluster_flow_list
            )

    return {
        "df_tmp": df_tmp,
        "rectangles": rectangles,
        "rectangle_colors": rectangle_colors,
        "country_flows": country_flows,
        "cluster_flows": cluster_flows,
        "year_list": year_list,
        "width": width,
        "total_share": total_share,
    }


def dyn_clusters_country_info(preprocessed, country_list):
    """
    Prepare information needed to highlight selected countries.

    This is intentionally cheap: all flow coordinates have already
    been calculated during preprocessing.
    """

    country_set = set(country_list)

    country_flows = preprocessed["country_flows"]
    cluster_flows = preprocessed["cluster_flows"]

    highlighted_flows = []
    highlighted_cluster_flows = []

    for flows in country_flows:
        if flows is None:
            highlighted_flows.append([])
            continue

        highlighted_flows.append([
            flow
            for flow in flows
            if flow[4] in country_set
        ])

    for flows in cluster_flows:
        if flows is None:
            highlighted_cluster_flows.append([])
            continue

        highlighted_cluster_flows.append([
            flow
            for flow in flows
            if flow[4] in country_set
        ])

    return {
        "country_list": list(country_list),
        "country_set": country_set,
        "country_flows": highlighted_flows,
        "cluster_flows": highlighted_cluster_flows,
    }


def dyn_clusters_draw_flow(
    ax,
    x0,
    y0,
    x1,
    y1,
    curvature=0.5,
    color="grey",
    alpha=0.5,
    zorder=0,
):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    dx = x1 - x0

    control1 = (x0 + curvature * dx, y0)
    control2 = (x1 - curvature * dx, y1)

    path = Path(
        [
            (x0, y0),
            control1,
            control2,
            (x1, y1),
        ],
        [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ],
    )

    patch = patches.PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_patch(patch)


def dyn_clusters_plot(
    preprocessed,
    country_info,
    base_fontsize=20,
    figsize=(24, 12),
    label_offsets = {
        "RU": 0,
        "ZA": 0,
        "ID": 0.02,
        "FR": -0.01,
        "CN": -0.01,
        "BR": -0.025,
    },
    ax= None,
):
    """
    Plot dynamic clusters and country flows.

    Parameters
    ----------
    preprocessed : dict
        Output of preprocess_dynamic_clusters().
    country_info : dict
        Output of prepare_country_info().

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    rectangles = preprocessed["rectangles"]
    rectangle_colors = preprocessed["rectangle_colors"]
    year_list_preproc = preprocessed["year_list"]

    country_flows = preprocessed["country_flows"]
    cluster_flows = preprocessed["cluster_flows"]

    country_set = country_info["country_set"]

    year_list = year_list_preproc[::-1]

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(
        "DYNAMIC CLUSTERS",
        fontsize=base_fontsize + 2,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # Cluster rectangles
    # ------------------------------------------------------------

    for (x, y, w, h), color in zip(
        rectangles,
        rectangle_colors,
    ):

        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            edgecolor=None,
            facecolor=color,
            zorder=1,
        )

        ax.add_patch(rect)

    # ------------------------------------------------------------
    # Country flows
    # ------------------------------------------------------------

    for flows in country_flows:

        if flows is None:
            continue

        for x0, y0, x1, y1, country in flows:

            if country in country_set:
                dyn_clusters_draw_flow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    curvature=0.4,
                    color="black",
                    alpha=0.8,
                    zorder=2,
                )

            else:
                dyn_clusters_draw_flow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    curvature=0.4,
                    color="lightgrey",
                    alpha=0.4,
                    zorder=2,
                )

    # ------------------------------------------------------------
    # Cluster flows + country labels
    # ------------------------------------------------------------

    printed_list = []

    for flows in cluster_flows:

        if flows is None:
            continue

        for x0, y0, x1, y1, country in flows:

            if country in country_set:

                dyn_clusters_draw_flow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    curvature=0.4,
                    color="black",
                    alpha=0.2,
                    zorder=2,
                )

                if country not in printed_list:

                    y_label = (
                        y1
                        + label_offsets.get(country, 0)
                    )

                    ax.text(
                        x1 + 0.08,
                        y_label,
                        country,
                        ha="left",
                        va="center",
                        fontsize=base_fontsize + 2,
                        fontweight="bold",
                    )

                    printed_list.append(country)

            else:

                dyn_clusters_draw_flow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    curvature=0.4,
                    color="white",
                    alpha=0.2,
                    zorder=2,
                )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    width = preprocessed["width"]

    ax.set_ylabel(
        "Share of countries by cluster, %",
        fontsize=base_fontsize,
    )

    ax.set_xlim(0, len(year_list))
    ax.set_ylim(0, 1)

    ax.set_xticks(
        np.arange(len(year_list)) + 1 - width / 2
    )

    ax.set_xticklabels(
        year_list,
        fontsize=base_fontsize,
    )

    ax.set_yticks(
        np.linspace(0, 1, 6),
        labels=np.linspace(0, 100, 6, dtype=int),
    )

    ax.tick_params(
        axis="y",
        labelsize=base_fontsize,
    )

    ax.tick_params(
        axis="x",
        color="none",
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=1,
        alpha=0.3,
        zorder=0,
    )

    ax.set_aspect("auto")

    return ax


def plot_clustering_scores(
    ax,
    year_list,
    silhouette_score,
    ari_score_mean,
    modularity,
    vertical_lines=None,
    ylim=(0, 1.05),
    color_list = ["#7d222e", "#b35f6a", "#9D9D9D"]
):
    """
    Plot clustering quality scores over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the plot.

    year_list : array-like
        Years.

    silhouette_score : array-like
        Silhouette scores.

    ari_score_mean : array-like
        Mean ARI scores.

    modularity : array-like
        Modularity scores.

    vertical_lines : array-like, optional
        Years at which to draw vertical reference lines.

    ylim : tuple, optional
        Y-axis limits.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    ax.plot(year_list, silhouette_score, color = color_list[0])
    ax.scatter(
        year_list,
        silhouette_score,
        label="silhouette",
        color = color_list[0]
    )

    ax.plot(year_list, ari_score_mean, color = color_list[1])
    ax.scatter(
        year_list,
        ari_score_mean,
        label="ARI",
        color = color_list[1]
    )

    ax.plot(year_list, modularity, color = color_list[2])
    ax.scatter(
        year_list,
        modularity,
        label="modularity",
        color = color_list[2]
    )

    if vertical_lines is not None:
        for year in vertical_lines:
            ax.axvline(year)

    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)

    ax.legend()

    return ax


def plot_cluster_map(
    ax,
    year,
    df_mapping,
    world,
    df_country,
    color_dict_tmp,
    title_fontsize=22,
):
    """
    Plot dynamic clusters on a world map for a given year.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the map.

    year : int
        Year to plot.

    df_mapping : pd.DataFrame
        Country-to-cluster mapping.

    world : geopandas.GeoDataFrame
        World geometries.

    df_country : pd.DataFrame
        Country code mapping containing 'alpha-2' and 'alpha-3'.

    color_dict_tmp : dict
        Mapping from dynamic cluster to color.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    # ------------------------------------------------------------
    # Merge country cluster information with country codes
    # ------------------------------------------------------------

    mapping_year = (
        df_mapping
        .query("year == @year")
        .merge(
            df_country[["alpha-2", "alpha-3"]],
            left_on="country",
            right_on="alpha-2",
            how="left",
        )
    )

    # ------------------------------------------------------------
    # Merge with world geometry
    # ------------------------------------------------------------

    world_ = world.merge(
        mapping_year,
        left_on="ISO_A3_EH",
        right_on="alpha-3",
        how="left",
    )

    # ------------------------------------------------------------
    # Map cluster → color
    # ------------------------------------------------------------

    world_["color"] = (
        world_["dynamic_cluster"]
        .map(color_dict_tmp)
        .fillna("white")
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    world_.plot(
        color=world_["color"],
        edgecolor="#242424",
        linewidth=0.6,
        ax=ax,
    )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    ax.set_axis_off()
    ax.margins(0)

    ax.set_title(
        year,
        fontsize=title_fontsize,
        fontweight="bold",
    )

    return ax
