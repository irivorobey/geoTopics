from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subfield_tree as sft


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data_clean"

df_subfield = (
    pd.read_csv(DATA_DIR / "df_topics.csv")
    [
        [
            "subfield_id",
            "subfield_name",
            "field_id",
            "field_name",
            "domain_id",
            "domain_name",
        ]
    ]
    .drop_duplicates()
    .sort_values(["domain_id", "field_id", "subfield_id"])
    .assign(subfield_id=lambda df: df["subfield_id"].astype(int))
    .reset_index(drop=True)
)

subfield_tree = sft.SubfieldTree(df_subfield)


DOMAIN_COLORS = {
    "Life Sciences": "#81b29a",
    "Physical Sciences": "#3d405b",
    "Social Sciences": "#e07a5f",
    "Health Sciences": "#f4f1de",
}


FIELD_NAME_DICT = {
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
    "Nursing": "Nursing",
}


def _white_to_color(color, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"white_to_{color}",
        ["white", color],
        N=n,
    )


def _build_field_colors(tree):
    field_colors = {}

    for domain, color in DOMAIN_COLORS.items():
        fields = (
            tree.df_subfield.loc[
                tree.df_subfield["domain_name"].eq(domain),
                "field_name",
            ]
            .drop_duplicates()
            .tolist()
        )

        cmap = _white_to_color(color)
        colors = cmap(np.linspace(0.2, 1, len(fields)))

        field_colors.update(
            zip(fields, [mcolors.to_hex(value) for value in colors])
        )

    return field_colors


FIELD_COLORS = _build_field_colors(subfield_tree)


def get_domain_pct(profile, tree, domain):
    topic_data = tree.df_subfield[["subfield_id", "domain_id"]]

    return int(
        profile.rename("profile")
        .rename_axis("subfield_id")
        .reset_index()
        .merge(topic_data, on="subfield_id")
        .loc[lambda df: df["domain_id"].eq(domain), "profile"]
        .sum()
        * 100
    )


def plot_two_profiles(
    df1,
    df2,
    subfield_tree,
    w1_tree,
    labels=None,
    title="Tree-Wasserstein distance",
    names=None,
):
    labels = labels or ["Profile 1", "Profile 2"]
    names = names or labels

    fig, ax = plt.subplots(figsize=(20, 5))
    x1 = df1.index.astype(str)
    x2 = df2.index.astype(str)

    for profile, x, color, label in [
        (df1, x1, "#1F6173", labels[0]),
        (df2, x2, "#D8370A", labels[1]),
    ]:
        ax.bar(x, profile.values, color=color, alpha=0.3)
        ax.scatter(x, profile.values, s=10, color=color, label=label)

    ticks = []
    tick_labels = []

    for i, domain in enumerate(subfield_tree.domain_list):
        first_subfield = (
            subfield_tree.df_subfield
            .query("domain_id == @domain")["subfield_id"]
            .iloc[0]
        )
        ticks.append(str(first_subfield))

        pct1 = get_domain_pct(df1, subfield_tree, domain)
        pct2 = get_domain_pct(df2, subfield_tree, domain)
        tick_labels.append(
            f"{subfield_tree.domain_name_list[i]}:\n"
            f"{labels[0]} = {pct1}%\n"
            f"{labels[1]} = {pct2}%"
        )
        ax.axvline(str(first_subfield), color="grey")

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        tick_labels,
        ha="left",
        fontsize=14,
    )

    for field in subfield_tree.field_list:
        first_subfield = (
            subfield_tree.df_subfield
            .query("field_id == @field")["subfield_id"]
            .iloc[0]
        )
        ax.axvline(
            str(first_subfield),
            color="grey",
            linestyle="--",
            alpha=0.2,
        )

    distance = w1_tree.dist(
        df1.to_frame(labels[0]).T,
        df2.to_frame(labels[1]).T,
    ).iloc[0, 0]

    ax.set_title(
        f"{names[0]} vs {names[1]}: "
        f"W-tree = {distance:.2f}",
        fontsize=16,
    )
    ax.set_ylabel("Subfield probability", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=14)

    return fig, ax


def plot_profile_pie_fields(
    df_profile,
    subfield_tree,
    plot_list=None,
    fig=None,
    gs=None,
    gs_idx=0,
    ax=None,
    title="Profiles",
):
    field_profiles = (
        df_profile
        .groupby(subfield_tree.field_name_dict, axis=1)
        .sum()
        .T
        .loc[subfield_tree.field_name_list]
    )

    plot_list = plot_list or field_profiles.columns.tolist()

    if fig is None:
        fig = plt.figure(figsize=(5 * len(plot_list), 5))

    if gs is None:
        gs = gridspec.GridSpec(1, 1, figure=fig)
        subspec = gs[0]
    else:
        subspec = gs[gs_idx, :]

    pie_grid = subspec.subgridspec(1, len(plot_list), wspace=0)
    axes = [
        fig.add_subplot(pie_grid[0, index])
        for index in range(len(plot_list))
    ]

    if ax is None:
        fig.suptitle(title, fontsize=18)
    else:
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")

    for axis, profile_name in zip(axes, plot_list):
        values = field_profiles[profile_name]
        colors = [
            FIELD_COLORS.get(field, "#cccccc")
            for field in subfield_tree.field_name_list
        ]

        axis.pie(
            values,
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else None,
            textprops={"fontsize": 14},
            pctdistance=0.8,
        )
        axis.set_xlabel(str(profile_name), fontsize=16)
        axis.set_aspect("equal")

    return fig, axes


def plot_field_legend(
    gs,
    subfield_tree,
    field_colors=None,
    field_name_dict=None,
):
    field_colors = field_colors or FIELD_COLORS
    field_name_dict = field_name_dict or FIELD_NAME_DICT

    axis = plt.subplot(gs[-1, 0])
    axis.axis("off")

    groups = defaultdict(list)
    for field in subfield_tree.field_name_list:
        domain = subfield_tree.field_domain_name_dict[field]
        groups[domain].append(field)

    for column, (domain, fields) in enumerate(groups.items()):
        x = 0.02 + column * 0.2
        axis.text(
            x,
            0.95,
            domain,
            fontsize=14,
            weight="bold",
            transform=axis.transAxes,
        )

        for row, field in enumerate(fields):
            y = 0.85 - row * 0.08

            axis.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.02),
                    0.02,
                    0.03,
                    color=field_colors.get(field, "#cccccc"),
                    transform=axis.transAxes,
                )
            )
            axis.text(
                x + 0.03,
                y,
                field_name_dict.get(field, field),
                va="center",
                fontsize=14,
                transform=axis.transAxes,
            )

    return axis