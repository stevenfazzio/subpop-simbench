"""Stage 5: build the interactive results page from reports/tables/.

    uv run python scripts/make_figures.py

Writes reports/figures/subpop_on_simbench.html: one page, three charts. Each chart is a set
of dumbbells running from a base model's SimBench score to its score with the SubPOP
adapter. A selector at the top switches all three between the two prompt formats that
cover the whole benchmark.
"""

import html
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from subpop_simbench.analysis import MODEL_PAIRS
from subpop_simbench.io import FIGURES_DIR, TABLES_DIR

FORMATS = {
    "simbench": "SimBench's own prompt format",
    "subpop_skeleton": "SubPOP's question layout",
}

# Scores from the SimBench paper's leaderboard (arXiv:2510.17516), a selection for context.
# All are for SimBench's own prompt format. Instruction-tuned models were asked to state a
# distribution in words; base models were read from first-token probabilities, as here.
PUBLISHED = [
    ("Claude-3.7-Sonnet", 40.80),
    ("GPT-4.1", 34.55),
    ("DeepSeek-R1", 34.52),
    ("Llama-3.1-405B-Instruct", 28.40),
    ("Qwen2.5-32B-Instruct", 23.76),
    ("Llama-3.1-70B-Instruct", 16.56),
    ("OLMo-2-0325-32B (best base model)", 15.90),
    ("Llama-3.1-Centaur-70B", 8.54),
    ("Llama-3.1-70B (base)", 1.21),
]

# One hue, two shades for before and after; grey for context. Validated with the dataviz
# palette checker as an ordinal ramp on each surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "base": "#86b6ef", "adapter": "#1c5cab", "published": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "base": "#184f95", "adapter": "#6da7ec", "published": "#898781",
    },
}  # fmt: skip
T = THEMES["light"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
SERIES = {
    "base": "Base model",
    "adapter": "With SubPOP adapter",
    "published": "Published on SimBench",
}
SUBSET_LABELS = {
    "all items": "All 13,510 items",
    "OpinionQA": "OpinionQA",
    "other opinion surveys": "Other opinion surveys",
    "non-survey tasks": "Non-survey tasks",
    "no matching training question": "No matching training question",
}
ROW_HEIGHT = 26


def _hover(title: str) -> str:
    return (
        f"<b>{title}</b><br>"
        "With adapter: <b>%{customdata[1]:.1f}</b><br>"
        "Base model: <b>%{customdata[0]:.1f}</b><br>"
        "Gain: %{customdata[2]:+.1f}<br>"
        "Items: %{customdata[3]:,}<extra></extra>"
    )


def add_dumbbells(fig, rows: pd.DataFrame, title: str, legend: bool, labels: bool, **where):
    """Draw base -> adapter for each row of `rows` (label, base, adapter, gain, n)."""
    xs, ys = [], []
    for row in rows.itertuples():
        xs += [row.base, row.adapter, None]
        ys += [row.label, row.label, None]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines", line={"color": T["axis"], "width": 2},
            hoverinfo="skip", showlegend=False, meta="connector",
        ),
        **where,
    )  # fmt: skip
    custom = rows[["base", "adapter", "gain", "n"]].to_numpy()
    for key in ("base", "adapter"):
        text = [f"{v:.1f}" for v in rows[key]] if labels and key == "adapter" else None
        fig.add_trace(
            go.Scatter(
                x=rows[key], y=rows["label"], name=SERIES[key], meta=key,
                mode="markers+text" if text else "markers",
                text=text, textposition="middle right",
                textfont={"color": T["ink"], "size": 12},
                marker={"color": T[key], "size": 11, "line": {"color": T["surface"], "width": 2}},
                customdata=custom, hovertemplate=_hover(title + "%{y}"),
                legendgroup=key, showlegend=legend, cliponaxis=False,
            ),
            **where,
        )  # fmt: skip


def style(fig, n_rows: int, x_range: tuple[float, float], left: int, order: list[str]):
    fig.update_layout(
        height=n_rows * ROW_HEIGHT + 96,
        margin={"l": left, "r": 28, "t": 62, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 13, "color": T["ink2"]},
        legend={"orientation": "h", "x": 0, "xref": "container", "xanchor": "left",
                "y": 1.0, "yref": "container", "yanchor": "top",
                "font": {"color": T["ink2"]}, "itemclick": False, "itemdoubleclick": False},
        hoverlabel={"bgcolor": T["surface"], "bordercolor": T["axis"],
                    "font": {"family": FONT, "color": T["ink"], "size": 13}},
        hovermode="closest", dragmode=False,
    )  # fmt: skip
    fig.update_xaxes(
        range=list(x_range), gridcolor=T["grid"], gridwidth=1, zeroline=True,
        zerolinecolor=T["axis"], zerolinewidth=1.5, tickfont={"color": T["muted"]},
        fixedrange=True, ticks="",
    )  # fmt: skip
    fig.update_yaxes(
        categoryorder="array", categoryarray=order[::-1], range=[-0.6, len(order) - 0.4],
        showgrid=False, zeroline=False, tickfont={"color": T["ink2"]}, fixedrange=True,
        ticks="",
    )  # fmt: skip
    for annotation in fig.layout.annotations:
        annotation.font = {"family": FONT, "size": 13, "color": T["ink"]}
    return fig


def x_range(*values, pad: float = 8.0) -> tuple[float, float]:
    flat = pd.concat([pd.Series(v, dtype=float) for v in values])
    return float(flat.min() - pad), float(flat.max() + pad + 6)


def headline_chart(overall: pd.DataFrame, condition: str, rng) -> tuple[go.Figure, pd.DataFrame]:
    ours = overall[overall["condition"] == condition].copy()
    ours["label"] = ours["model_name"] + " + SubPOP"
    ours = ours.sort_values("adapter", ascending=False)
    table = ours[["label", "base", "adapter", "gain", "n"]].copy()
    table["published"] = float("nan")
    fig = go.Figure()
    order = list(ours["label"])
    add_dumbbells(fig, ours, "", legend=True, labels=True)
    if condition == "simbench":
        pub = pd.DataFrame(PUBLISHED, columns=["label", "published"])
        pub["label"] += " · published"
        table = pd.concat([table, pub], ignore_index=True)
        table["rank"] = table["adapter"].fillna(table["published"])
        table = table.sort_values("rank", ascending=False).drop(columns="rank")
        order = list(table["label"])
        fig.add_trace(
            go.Scatter(
                x=pub["published"], y=pub["label"], name=SERIES["published"], meta="published",
                mode="markers+text", text=[f"{v:.1f}" for v in pub["published"]],
                textposition="middle right", textfont={"color": T["ink"], "size": 12},
                marker={"color": T["published"], "size": 10, "symbol": "diamond",
                        "line": {"color": T["surface"], "width": 2}},
                hovertemplate="<b>%{y}</b><br>Published score: <b>%{x:.2f}</b><extra></extra>",
                cliponaxis=False,
            )
        )  # fmt: skip
    return style(fig, len(order), rng, left=250, order=order), table.reset_index(drop=True)


def panel_chart(rows: pd.DataFrame, order: list[str], rng, left: int) -> go.Figure:
    """Small multiples: one panel per model, the same rows and the same scale in each."""
    models = [name for _, name in MODEL_PAIRS.values() if name in set(rows["model_name"])]
    fig = make_subplots(
        rows=1, cols=len(models), shared_yaxes=True, subplot_titles=models,
        horizontal_spacing=0.025,
    )  # fmt: skip
    for col, name in enumerate(models, start=1):
        sub = rows[rows["model_name"] == name].set_index("label").loc[order].reset_index()
        add_dumbbells(fig, sub, f"{name} · ", legend=col == 1, labels=False, row=1, col=col)
    return style(fig, len(order), rng, left=left, order=order)


def table_html(df: pd.DataFrame, columns: dict[str, str]) -> str:
    """A plain HTML table: the chart's values without needing to hover."""
    numeric = ' class="num"'
    head = "".join(
        f"<th{numeric if i else ''}>{html.escape(title)}</th>"
        for i, title in enumerate(columns.values())
    )
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for column in columns:
            value = getattr(row, column)
            if isinstance(value, str):
                cells.append(f"<td>{html.escape(value)}</td>")
            elif pd.isna(value):
                cells.append('<td class="num"></td>')
            elif column == "n":
                cells.append(f'<td class="num">{int(value):,}</td>')
            else:
                cells.append(f'<td class="num">{value:.1f}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def wide_table(rows: pd.DataFrame, order: list[str], first: str) -> str:
    """Rows down, models across, 'base → adapter' in each cell."""
    models = [name for _, name in MODEL_PAIRS.values() if name in set(rows["model_name"])]
    head = f'<th>{html.escape(first)}</th><th class="num">Items</th>' + "".join(
        f'<th class="num">{html.escape(m)}</th>' for m in models
    )
    indexed = rows.set_index(["label", "model_name"])
    body = []
    for label in order:
        n = int(indexed.loc[(label, models[0]), "n"])
        cells = "".join(
            f'<td class="num">{indexed.loc[(label, m), "base"]:.1f} → '
            f"{indexed.loc[(label, m), 'adapter']:.1f}</td>"
            for m in models
        )
        body.append(f'<tr><td>{html.escape(label)}</td><td class="num">{n:,}</td>{cells}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def chart_block(fig: go.Figure, div_id: str, table: str, condition: str, wide: bool) -> str:
    """A chart and its table. On a narrow screen the chart scrolls sideways inside its card."""
    plot = fig.to_html(
        full_html=False, include_plotlyjs=False, div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )  # fmt: skip
    size = "wide" if wide else "mid"
    return (
        f'<div class="fmt fmt-{condition}"><div class="scroll"><div class="plot {size}">'
        f"{plot}</div></div><details><summary>Show as a table</summary>{table}</details></div>"
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SubPOP adapters on SimBench</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    --surface: {light[surface]}; --page: #f9f9f7; --ink: {light[ink]}; --ink2: {light[ink2]};
    --muted: {light[muted]}; --hairline: {light[grid]}; --base: {light[base]};
    --adapter: {light[adapter]}; --published: {light[published]};
    margin: 0; background: var(--page); color: var(--ink);
    font: 15px/1.5 {font};
  }}
  @media (prefers-color-scheme: dark) {{
    body {{
      --surface: {dark[surface]}; --page: #0d0d0d; --ink: {dark[ink]}; --ink2: {dark[ink2]};
      --muted: {dark[muted]}; --hairline: {dark[grid]}; --base: {dark[base]};
      --adapter: {dark[adapter]}; --published: {dark[published]};
    }}
  }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 28px 16px 56px; }}
  h1 {{ font-size: 26px; line-height: 1.2; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 0 0 2px; }}
  p {{ margin: 0 0 10px; max-width: 76ch; }}
  .lede {{ color: var(--ink2); }}
  .sub {{ color: var(--ink2); font-size: 14px; }}
  section {{
    background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px;
    padding: 18px 18px 10px; margin: 18px 0;
  }}
  .filters {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 18px 0 0;
    font-size: 14px; color: var(--ink2);
  }}
  .filters button {{
    font: inherit; color: var(--ink2); background: transparent; cursor: pointer;
    border: 1px solid var(--hairline); border-radius: 999px; padding: 5px 14px;
  }}
  .filters button[aria-pressed="true"] {{
    color: var(--surface); background: var(--ink); border-color: var(--ink);
  }}
  .filters button:focus-visible {{ outline: 2px solid var(--adapter); outline-offset: 2px; }}
  body[data-format="simbench"] .fmt-subpop_skeleton,
  body[data-format="subpop_skeleton"] .fmt-simbench {{ display: none; }}
  .scroll {{ overflow-x: auto; }}
  .plot.wide {{ min-width: 900px; }}
  .plot.mid {{ min-width: 620px; }}
  details {{ margin: 2px 0 10px; font-size: 13.5px; }}
  summary {{ cursor: pointer; color: var(--ink2); padding: 4px 0; }}
  details > table {{ display: block; overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: 6px 0 4px; }}
  th, td {{ text-align: left; padding: 4px 14px 4px 0; border-bottom: 1px solid var(--hairline); }}
  th {{ color: var(--ink2); font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }}
  th.num {{ text-align: right; }}
  footer {{ color: var(--ink2); font-size: 13.5px; }}
  footer li {{ margin-bottom: 6px; max-width: 80ch; }}
  a {{ color: inherit; }}
</style>
<script src="https://cdn.plot.ly/plotly-{plotlyjs}.min.js" charset="utf-8"></script>
</head>
<body data-format="simbench">
<main>
<h1>SubPOP adapters on SimBench</h1>
<p class="lede">Four language models, each scored on SimBench with and without its SubPOP
adapter, on identical prompts. Every dumbbell runs from the base model's score to the score
with the adapter. A score of 100 matches the human answer distribution; 0 is no better than
answering uniformly; below 0 is worse than uniform.</p>

<div class="filters" role="group" aria-label="Prompt format">
  <span>Prompt format</span>
  {buttons}
</div>

<section>
  <h2>Overall score</h2>
  <p class="sub">All 13,510 SimBench items. {headline_note}</p>
  {headline}
</section>

<section>
  <h2>By kind of dataset</h2>
  <p class="sub">The gain is largest on OpinionQA, the survey family SubPOP trained on, and
  holds on other surveys, on non-survey tasks, and on the 12,564 items whose question matches
  nothing in SubPOP's training data.</p>
  {subsets}
</section>

<section>
  <h2>By dataset</h2>
  <p class="sub">All 25 dataset and split combinations, ordered by average gain within each
  split. Some datasets are small: the table gives item counts.</p>
  {datasets}
</section>

<footer>
<ul>
  <li>Scores are means over items of 100 × (1 − TVD / norm), where TVD is the total
  variation distance between the model's and the humans' answer distributions and norm is
  the dataset's mean TVD between humans and uniform.</li>
  <li>Model answers are read from next-token probabilities over the option letters, in
  bfloat16, one forward pass per prompt.</li>
  <li>Published scores are from the SimBench paper
  (<a href="https://arxiv.org/abs/2510.17516">arXiv:2510.17516</a>) and apply to SimBench's
  own prompt format only. Its instruction-tuned models were asked to state a distribution
  in words, and its open-weight base models were run 8-bit or 4-bit quantised, so the
  comparison is indicative.</li>
  <li>On SubPOP's own prompt format, which exists for 964 OpinionQA items, the adapters
  score {exact}.</li>
</ul>
</footer>
</main>
<script>
const THEMES = {themes};
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme() {{
  const t = darkQuery.matches ? THEMES.dark : THEMES.light;
  document.querySelectorAll(".js-plotly-plot").forEach((gd) => {{
    const layout = {{
      "font.color": t.ink2, "legend.font.color": t.ink2, "hoverlabel.bgcolor": t.surface,
      "hoverlabel.bordercolor": t.axis, "hoverlabel.font.color": t.ink,
    }};
    Object.keys(gd.layout).forEach((key) => {{
      if (key.startsWith("xaxis")) {{
        layout[key + ".gridcolor"] = t.grid;
        layout[key + ".zerolinecolor"] = t.axis;
        layout[key + ".tickfont.color"] = t.muted;
      }} else if (key.startsWith("yaxis")) {{
        layout[key + ".tickfont.color"] = t.ink2;
      }}
    }});
    (gd.layout.annotations || []).forEach((_, i) => {{
      layout["annotations[" + i + "].font.color"] = t.ink;
    }});
    const byKind = {{}};
    gd.data.forEach((trace, i) => (byKind[trace.meta] = byKind[trace.meta] || []).push(i));
    Object.entries(byKind).forEach(([kind, indices]) => {{
      const update = kind === "connector"
        ? {{"line.color": t.axis}}
        : {{"marker.color": t[kind], "marker.line.color": t.surface, "textfont.color": t.ink}};
      Plotly.restyle(gd, update, indices);
    }});
    Plotly.relayout(gd, layout);
  }});
}}

function setFormat(format) {{
  document.body.dataset.format = format;
  document.querySelectorAll(".filters button").forEach((button) => {{
    button.setAttribute("aria-pressed", String(button.dataset.format === format));
  }});
  document.querySelectorAll(".fmt-" + format + " .js-plotly-plot").forEach((gd) => {{
    Plotly.Plots.resize(gd);
  }});
}}

document.querySelectorAll(".filters button").forEach((button) => {{
  button.addEventListener("click", () => setFormat(button.dataset.format));
}});
window.addEventListener("load", () => {{
  if (darkQuery.matches) applyTheme();
  darkQuery.addEventListener("change", applyTheme);
}});
</script>
</body>
</html>
"""


def main() -> None:
    overall = pd.read_csv(TABLES_DIR / "overall.csv")
    subsets = pd.read_csv(TABLES_DIR / "by_subset.csv")
    datasets = pd.read_csv(TABLES_DIR / "by_dataset.csv")

    both = overall[overall["condition"].isin(FORMATS)]
    head_range = x_range(both["base"], both["adapter"], [p[1] for p in PUBLISHED])

    subsets = subsets[subsets["subset"].isin(SUBSET_LABELS) & subsets["condition"].isin(FORMATS)]
    subsets = subsets.assign(label=subsets["subset"].map(SUBSET_LABELS))
    subset_order = [v for k, v in SUBSET_LABELS.items() if k in set(subsets["subset"])]
    subset_range = x_range(subsets["base"], subsets["adapter"])

    datasets = datasets[datasets["condition"].isin(FORMATS)].copy()
    split_names = {"pop": "Pop", "grouped": "Grouped"}
    datasets["label"] = datasets["dataset_name"] + " (" + datasets["split"].map(split_names) + ")"
    # One fixed order for both formats, so switching format never reorders the rows.
    mean_gain = datasets.groupby(["split", "label"])["gain"].mean().reset_index()
    mean_gain["split_rank"] = mean_gain["split"].map({"pop": 0, "grouped": 1})
    dataset_order = list(
        mean_gain.sort_values(["split_rank", "gain"], ascending=[True, False])["label"]
    )
    dataset_range = x_range(datasets["base"], datasets["adapter"])

    blocks = {"headline": [], "subsets": [], "datasets": []}
    for condition in FORMATS:
        fig, table = headline_chart(overall, condition, head_range)
        columns = {"label": "Model", "base": "Base", "adapter": "With adapter", "gain": "Gain"}
        if condition == "simbench":
            columns["published"] = "Published"
        blocks["headline"].append(
            chart_block(fig, f"headline-{condition}", table_html(table, columns), condition, False)
        )
        sub = subsets[subsets["condition"] == condition]
        blocks["subsets"].append(
            chart_block(
                panel_chart(sub, subset_order, subset_range, left=215),
                f"subsets-{condition}",
                wide_table(sub, subset_order, "Items scored"),
                condition,
                True,
            )
        )
        sub = datasets[datasets["condition"] == condition]
        blocks["datasets"].append(
            chart_block(
                panel_chart(sub, dataset_order, dataset_range, left=215),
                f"datasets-{condition}",
                wide_table(sub, dataset_order, "Dataset (split)"),
                condition,
                True,
            )
        )

    exact = overall[overall["condition"] == "subpop_exact"]
    exact_text = "; ".join(
        f"{row.model_name} {row.adapter:.1f} (base {row.base:.1f})" for row in exact.itertuples()
    )
    buttons = "\n  ".join(
        f'<button type="button" data-format="{key}" aria-pressed="{str(i == 0).lower()}">'
        f"{html.escape(label)}</button>"
        for i, (key, label) in enumerate(FORMATS.items())
    )
    page = PAGE.format(
        light=THEMES["light"],
        dark=THEMES["dark"],
        font=FONT,
        plotlyjs=get_plotlyjs_version(),
        buttons=buttons,
        headline_note=(
            "Grey diamonds are scores published in the SimBench paper, shown for SimBench's "
            "own prompt format only."
        ),
        headline="\n".join(blocks["headline"]),
        subsets="\n".join(blocks["subsets"]),
        datasets="\n".join(blocks["datasets"]),
        exact=html.escape(exact_text),
        themes=json.dumps(THEMES),
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "subpop_on_simbench.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
