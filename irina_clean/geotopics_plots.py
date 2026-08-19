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