"""Stage 5: build the interactive results page from reports/tables/.

    uv run python scripts/make_figures.py

Writes reports/figures/subpop_on_simbench.html: one page. Its first four charts are sets of
dumbbells running from a base model's SimBench score to its score with the SubPOP adapter,
next to what a predictor that never reads the question scores. A selector at the top
switches all four between the two prompt formats that cover the whole benchmark.

A last section, drawn when reports/tables/retrain.csv exists, sets the adapter retrained
here beside the released one under all three prompt formats, and shows its training and
validation loss per epoch.
"""

import html
import json
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from subpop_simbench.analysis import MODEL_PAIRS, RELEASED, RETRAINED
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
        "base": "#86b6ef", "adapter": "#1c5cab", "published": "#898781", "blind": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "base": "#184f95", "adapter": "#6da7ec", "published": "#898781", "blind": "#c3c2b7",
    },
}  # fmt: skip
T = THEMES["light"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LEGEND_RANK = {"base": 1, "adapter": 2, "published": 3, "blind": 4}
SERIES = {
    "base": "Base model",
    "adapter": "With SubPOP adapter",
    "published": "Published on SimBench",
    "blind": "Never reads the question",
}
BLIND_LABELS = {
    "dataset": "Question-blind: each dataset's typical shape",
    "benchmark": "Question-blind: one shape per option count",
}
SUBSET_LABELS = {
    "all items": "All 13,510 items",
    "OpinionQA": "OpinionQA",
    "other opinion surveys": "Other opinion surveys",
    "non-survey tasks": "Non-survey tasks",
    "no matching training question": "No matching training question",
}
ROW_HEIGHT = 26
# The retraining section shows every format at once, including SubPOP's own.
RETRAIN_FORMATS = {**FORMATS, "subpop_exact": "SubPOP's exact format (964 items)"}
RETRAIN_SHORT = {
    "simbench": "SimBench's own",
    "subpop_skeleton": "SubPOP's layout",
    "subpop_exact": "SubPOP's exact",
}


def _hover(title: str) -> str:
    return (
        f"<b>{title}</b><br>"
        "With adapter: <b>%{customdata[1]:.1f}</b><br>"
        "Base model: <b>%{customdata[0]:.1f}</b><br>"
        "Gain: %{customdata[2]:+.1f}<br>"
        "Never reads the question: %{customdata[4]:.1f}<br>"
        "Items: %{customdata[3]:,}<extra></extra>"
    )


def add_dumbbells(
    fig, rows: pd.DataFrame, title: str, legend: bool, labels: bool, ticks: bool = True, **where
):
    """Draw base -> adapter for each row of `rows` (label, base, adapter, gain, n).

    With `ticks`, a vertical tick marks `question_blind` on each row: the score, on the same
    items, of predicting the dataset's typical answer shape without reading the question.
    """
    if ticks:
        fig.add_trace(
            go.Scatter(
                x=rows["question_blind"], y=rows["label"], name=SERIES["blind"],
                meta="blindtick", mode="markers",
                marker={"symbol": "line-ns", "size": 13, "color": T["blind"],
                        "line": {"color": T["blind"], "width": 2}},
                hovertemplate=f"<b>{title}%{{y}}</b><br>Never reads the question: "
                "<b>%{x:.1f}</b><extra></extra>",
                legendgroup="blind", showlegend=legend, legendrank=LEGEND_RANK["blind"],
            ),
            **where,
        )  # fmt: skip
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
    custom = rows[["base", "adapter", "gain", "n", "question_blind"]].to_numpy()
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
                legendgroup=key, showlegend=legend, legendrank=LEGEND_RANK[key],
                cliponaxis=False,
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


def reference_trace(rows: pd.DataFrame, kind: str, symbol: str, hover: str) -> go.Scatter:
    """Single-value rows on the headline chart: published scores and question-blind baselines."""
    return go.Scatter(
        x=rows["reference"], y=rows["label"], name=SERIES[kind], meta=kind,
        mode="markers+text", text=[f"{v:.1f}" for v in rows["reference"]],
        textposition="middle right", textfont={"color": T["ink"], "size": 12},
        marker={"color": T[kind], "size": 10, "symbol": symbol,
                "line": {"color": T["surface"], "width": 2}},
        hovertemplate=f"<b>%{{y}}</b><br>{hover}: <b>%{{x:.2f}}</b><extra></extra>",
        legendrank=LEGEND_RANK[kind], cliponaxis=False,
    )  # fmt: skip


def headline_chart(
    overall: pd.DataFrame, blind: pd.DataFrame, condition: str, rng
) -> tuple[go.Figure, pd.DataFrame]:
    ours = overall[overall["condition"] == condition].copy()
    ours["label"] = ours["model_name"] + " + SubPOP"
    table = ours[["label", "base", "adapter", "gain", "n"]].copy()
    fig = go.Figure()
    add_dumbbells(fig, ours, "", legend=True, labels=True, ticks=False)
    references = []
    if condition == "simbench":
        pub = pd.DataFrame(PUBLISHED, columns=["label", "reference"])
        pub["label"] += " · published"
        fig.add_trace(reference_trace(pub, "published", "diamond", "Published score"))
        references.append(pub)
    # The baselines never see a prompt, so they are the same under either format.
    base = blind[blind["subset"] == "all items"]
    base = pd.DataFrame({"label": base["baseline"].map(BLIND_LABELS), "reference": base["score"]})
    fig.add_trace(reference_trace(base, "blind", "square", "Score without reading the question"))
    references.append(base)
    table = pd.concat([table, *references], ignore_index=True)
    table["rank"] = table["adapter"].fillna(table["reference"])
    table = table.sort_values("rank", ascending=False).drop(columns="rank")
    order = list(table["label"])
    return style(fig, len(order), rng, left=300, order=order), table.reset_index(drop=True)


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


def recalibrated_chart(controls: pd.DataFrame, overall: pd.DataFrame, condition: str, rng):
    """Base and adapter after the same generic correction: letter usage and one temperature."""
    rows = controls[controls["condition"] == condition].merge(
        overall[["model", "condition", "n", "question_blind"]], on=["model", "condition"]
    )
    rows = pd.DataFrame(
        {
            "label": rows["model_name"],
            "raw_base": rows["base"],
            "base": rows["base_letter_usage_matched"],
            "raw_adapter": rows["adapter"],
            "adapter": rows["adapter_letter_usage_matched"],
            "gain": rows["adapter_letter_usage_matched"] - rows["base_letter_usage_matched"],
            "n": rows["n"],
            "question_blind": rows["question_blind"],
        }
    )
    fig = go.Figure()
    add_dumbbells(fig, rows, "", legend=True, labels=True, ticks=False)
    reference = pd.DataFrame(
        {"label": [BLIND_LABELS["dataset"]], "reference": [rows["question_blind"].iloc[0]]}
    )
    fig.add_trace(
        reference_trace(reference, "blind", "square", "Score without reading the question")
    )
    table = pd.concat([rows, reference], ignore_index=True)
    table["rank"] = table["adapter"].fillna(table["reference"])
    table = table.sort_values("rank", ascending=False).drop(columns="rank")
    order = list(table["label"])
    return style(fig, len(order), rng, left=300, order=order), table.reset_index(drop=True)


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
    head = (
        f'<th>{html.escape(first)}</th><th class="num">Items</th>'
        '<th class="num">Never reads the question</th>'
        + "".join(f'<th class="num">{html.escape(m)}</th>' for m in models)
    )
    indexed = rows.set_index(["label", "model_name"])
    body = []
    for label in order:
        n = int(indexed.loc[(label, models[0]), "n"])
        blind = indexed.loc[(label, models[0]), "question_blind"]
        cells = f'<td class="num">{blind:.1f}</td>' + "".join(
            f'<td class="num">{indexed.loc[(label, m), "base"]:.1f} → '
            f"{indexed.loc[(label, m), 'adapter']:.1f}</td>"
            for m in models
        )
        body.append(f'<tr><td>{html.escape(label)}</td><td class="num">{n:,}</td>{cells}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def chart_block(fig: go.Figure, div_id: str, table: str, condition: str | None, wide: bool) -> str:
    """A chart and its table. On a narrow screen the chart scrolls sideways inside its card.

    With a `condition` the block shows only while that prompt format is selected.
    """
    plot = fig.to_html(
        full_html=False, include_plotlyjs=False, div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )  # fmt: skip
    size = "wide" if wide else "mid"
    classes = f"fmt fmt-{condition}" if condition else "always"
    return (
        f'<div class="{classes}"><div class="scroll"><div class="plot {size}">'
        f"{plot}</div></div><details><summary>Show as a table</summary>{table}</details></div>"
    )


def retrain_rows(retrain: pd.DataFrame, overall: pd.DataFrame, model: str) -> pd.DataFrame:
    """Dumbbell rows per prompt format: the released adapter, then each retrained checkpoint."""
    rows = retrain[
        (retrain["model"] == model)
        & (retrain["subset"] == "all items")
        & (retrain["reference"] == RELEASED)
    ]
    blind = overall[overall["model"] == model].set_index("condition")["question_blind"]
    out = []
    for condition, group in rows.groupby("condition", sort=False):
        first = group.iloc[0]
        shared = {"condition": condition, "n": first["n"], "base": first["base"]}
        out.append({**shared, "label": "Released adapter", "adapter": first["reference_score"]})
        for row in group.itertuples():
            kind = "lowest validation loss" if row.run == "best validation loss" else "final"
            out.append(
                {
                    **shared,
                    "label": f"Retrained, epoch {row.epoch} ({kind})",
                    "adapter": row.score,
                    "difference": row.difference,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                }
            )
    out = pd.DataFrame(out)
    out["gain"] = out["adapter"] - out["base"]
    out["question_blind"] = out["condition"].map(blind)
    return out


def retrain_chart(rows: pd.DataFrame, order: list[str], rng) -> go.Figure:
    """One panel per prompt format, the same three adapters and the same scale in each."""
    formats = [c for c in RETRAIN_FORMATS if c in set(rows["condition"])]
    fig = make_subplots(
        rows=1, cols=len(formats), shared_yaxes=True, horizontal_spacing=0.025,
        subplot_titles=[RETRAIN_FORMATS[c] for c in formats],
    )  # fmt: skip
    for col, condition in enumerate(formats, start=1):
        sub = rows[rows["condition"] == condition].set_index("label").loc[order].reset_index()
        title = f"{RETRAIN_SHORT[condition]} format · "
        add_dumbbells(fig, sub, title, legend=col == 1, labels=True, row=1, col=col)
    return style(fig, len(order), rng, left=300, order=order)


def retrain_table(rows: pd.DataFrame, order: list[str]) -> str:
    """Formats down, adapters across, each retrained score with its gap to the released one."""
    head = '<th>Prompt format</th><th class="num">Items</th><th class="num">Base model</th>'
    for label in order:
        head += f'<th class="num">{html.escape(label)}</th>'
        if label != order[0]:
            head += '<th class="num">Minus released (95% interval)</th>'
    body = []
    indexed = rows.set_index(["condition", "label"])
    for condition in [c for c in RETRAIN_FORMATS if c in set(rows["condition"])]:
        first = indexed.loc[(condition, order[0])]
        cells = (
            f"<td>{html.escape(RETRAIN_FORMATS[condition])}</td>"
            f'<td class="num">{int(first["n"]):,}</td><td class="num">{first["base"]:.1f}</td>'
        )
        for label in order:
            row = indexed.loc[(condition, label)]
            cells += f'<td class="num">{row["adapter"]:.1f}</td>'
            if label != order[0]:
                cells += (
                    f'<td class="num">{row["difference"]:+.1f} '
                    f"({row['ci_low']:+.1f} to {row['ci_high']:+.1f})</td>"
                )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def loss_chart(epochs: pd.DataFrame, references: dict[str, float]) -> go.Figure:
    """Training and held-out loss per epoch, over the losses of predictors that need no model.

    Both lines are the retrained adapter, so both wear its colour: solid for the questions
    held out, dotted for the ones trained on. The larger rings are the epochs scored on
    SimBench.
    """
    last = epochs.iloc[-1]
    best = epochs[epochs["lowest_val_loss"]].iloc[0]
    scored = epochs["lowest_val_loss"] | (epochs["epoch"] == last["epoch"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs["epoch"], y=epochs["train_loss"], name="Questions trained on",
            meta="loss", mode="lines", line={"color": T["adapter"], "width": 2, "dash": "dot"},
            hovertemplate="Questions trained on: <b>%{y:.4f}</b><extra></extra>",
        )
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=epochs["epoch"], y=epochs["val_loss"], name="Questions held out",
            meta="loss", mode="lines+markers", line={"color": T["adapter"], "width": 2},
            marker={"color": T["adapter"], "size": [12 if s else 7 for s in scored],
                    "line": {"color": T["surface"], "width": 2}},
            hovertemplate="Questions held out: <b>%{y:.4f}</b><extra></extra>",
        )
    )  # fmt: skip
    note = {"family": FONT, "size": 12, "color": T["ink2"]}
    for label, value in references.items():
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, y0=value, y1=value, layer="below",
            line={"color": T["muted"], "width": 1, "dash": "dash"},
        )  # fmt: skip
        below = value < epochs["train_loss"].min()
        fig.add_annotation(
            xref="paper", x=1, y=value, text=f"{label}: {value:.4f}", showarrow=False,
            xanchor="right", yanchor="top" if below else "bottom", font=note, name="note",
        )  # fmt: skip
    fig.add_annotation(
        x=best["epoch"], y=best["val_loss"], yshift=-12, showarrow=False, yanchor="top",
        text=f"Lowest: epoch {int(best['epoch'])}, {best['val_loss']:.4f}", font=note,
        name="note",
    )  # fmt: skip
    top = math.ceil(max(epochs["train_loss"].max(), epochs["val_loss"].max()) * 50) / 50
    fig.update_layout(
        height=380,
        margin={"l": 56, "r": 28, "t": 62, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 13, "color": T["ink2"]},
        legend={"orientation": "h", "x": 0, "xref": "container", "xanchor": "left",
                "y": 1.0, "yref": "container", "yanchor": "top",
                "font": {"color": T["ink2"]}, "itemclick": False, "itemdoubleclick": False},
        hoverlabel={"bgcolor": T["surface"], "bordercolor": T["axis"],
                    "font": {"family": FONT, "color": T["ink"], "size": 13}},
        hovermode="x unified", dragmode=False,
    )  # fmt: skip
    fig.update_xaxes(
        title={"text": "Epoch", "standoff": 8}, range=[0.4, last["epoch"] + 0.6], dtick=1,
        showgrid=False, zeroline=False, tickfont={"color": T["muted"]}, fixedrange=True,
        ticks="", hoverformat="d", tickprefix="", unifiedhovertitle={"text": "Epoch %{x}"},
    )  # fmt: skip
    fig.update_yaxes(
        title={"text": "Loss", "standoff": 8}, range=[0, top], gridcolor=T["grid"],
        gridwidth=1, zeroline=True, zerolinecolor=T["axis"], zerolinewidth=1.5,
        tickfont={"color": T["ink2"]}, tickformat=".2f", fixedrange=True, ticks="",
    )  # fmt: skip
    return fig


def epochs_table(epochs: pd.DataFrame) -> str:
    head = (
        '<th class="num">Epoch</th><th class="num">Loss, questions trained on</th>'
        '<th class="num">Loss, questions held out</th>'
    )
    body = "".join(
        f'<tr><td class="num">{int(row.epoch)}</td><td class="num">{row.train_loss:.4f}</td>'
        f'<td class="num">{row.val_loss:.4f}{" (lowest)" if row.lowest_val_loss else ""}</td></tr>'
        for row in epochs.itertuples()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


RETRAIN_SECTION = """
<section>
  <h2>Retraining the {name} adapter</h2>
  <p class="sub">SubPOP's released adapters fit only their own four base models. As a check
  that the recipe can be rerun, the {name} adapter was trained again here from SubPOP's
  data, for {n_epochs} epochs. Two checkpoints were scored: epoch {best_epoch}, which had the
  lowest loss on held-out questions, and the final one. Both land within {max_gap} points of
  the released adapter under every prompt format, and on {closer_low} to {closer_high} of
  items their predictions are nearer the released adapter's than the base model's. The
  format selector does not apply here: the panels are the three formats.</p>
  {chart}
  <p class="sub">Loss per epoch on the questions trained on and on those held out ({held_out};
  lower is better). The larger rings are the two epochs scored above. The held-out loss
  stops falling after a few epochs while the training loss keeps going, yet the final epoch
  scores {exact_gain} points higher than epoch {best_epoch} on SubPOP's exact format. The
  released adapter's {released_loss} on these held-out questions is not a target to compare
  with: it is below the {question_average} that comes from knowing each question's average
  answer, so it has probably trained on most of them.</p>
  {loss}
</section>
"""


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
<p class="lede">For scale: a predictor that never reads the question, and knows only each
dataset's typical answer shape, scores {blind_dataset}. Squares and vertical ticks mark
that baseline.</p>

<div class="filters" role="group" aria-label="Prompt format">
  <span>Prompt format</span>
  {buttons}
</div>

<section>
  <h2>Overall score</h2>
  <p class="sub">All 13,510 SimBench items. Grey diamonds are scores published in the
  SimBench paper, shown for SimBench's own prompt format only. Squares are predictors that
  never read the question.</p>
  {headline}
</section>

<section>
  <h2>After the same generic correction</h2>
  <p class="sub">Base models often favour particular option letters, which costs score for
  reasons unrelated to what they know. Here the base model and the adapter each have their
  average letter usage shifted to the human average and one temperature applied. On this
  footing every adapter is {recal_low} to {recal_high} points above its base model. The
  square is the question-blind score.</p>
  {recalibrated}
</section>

<section>
  <h2>By kind of dataset</h2>
  <p class="sub">The gain is largest on OpinionQA, the survey family SubPOP trained on, and
  holds on other surveys, on non-survey tasks, and on the 12,564 items whose question matches
  nothing in SubPOP's training data. The tick is the question-blind score on the same items:
  on OpinionQA it is already high, and on non-survey tasks only the 70B adapter passes it.</p>
  {subsets}
</section>

<section>
  <h2>By dataset</h2>
  <p class="sub">All 25 dataset and split combinations, ordered by average gain within each
  split. Some datasets are small: the table gives item counts. Ticks as above. The
  MoralMachine row should not be read as a result: its human answers look reversed in
  SimBench's data (see the notes at the foot of the page).</p>
  {datasets}
</section>
{retrain}
<footer>
<ul>
  <li>Scores are means over items of 100 × (1 − TVD / norm), where TVD is the total
  variation distance between the model's and the humans' answer distributions and norm is
  the dataset's mean TVD between humans and uniform.</li>
  <li>Model answers are read from next-token probabilities over the option letters, in
  bfloat16, one forward pass per prompt.</li>
  <li>The question-blind predictors use human answers to other questions: the average
  answer shape of every other question with the same number of options, either in the
  item's own dataset ({blind_dataset}) or across the whole benchmark ({blind_benchmark}). No
  model gets these for free; they show how much score answer shape alone is worth.</li>
  <li>Published scores are from the SimBench paper
  (<a href="https://arxiv.org/abs/2510.17516">arXiv:2510.17516</a>) and apply to SimBench's
  own prompt format only. Its instruction-tuned models were asked to state a distribution
  in words, and its open-weight base models were run 8-bit or 4-bit quantised, so the
  comparison is indicative.</li>
  <li>SimBench's human answers for its 500 MoralMachine items look reversed
  (<a href="https://github.com/pitehu/SimBench_release/issues/2">open issue on SimBench's
  repository</a>). In the {mm_animals} items that pit only animals against only people, the
  majority is recorded as choosing to kill the people in {mm_killed}. This page scores the
  data as published. With those answers reversed, no overall score here moves by more than
  {mm_shift} points and no adapter scores above {mm_best} on MoralMachine. If the published
  scores were computed on the same data, they may move by more.</li>
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
        layout[key + ".gridcolor"] = t.grid;
        layout[key + ".zerolinecolor"] = t.axis;
      }}
    }});
    (gd.layout.annotations || []).forEach((annotation, i) => {{
      layout["annotations[" + i + "].font.color"] = annotation.name === "note" ? t.ink2 : t.ink;
    }});
    const byKind = {{}};
    gd.data.forEach((trace, i) => (byKind[trace.meta] = byKind[trace.meta] || []).push(i));
    Object.entries(byKind).forEach(([kind, indices]) => {{
      let update = {{
        "marker.color": t[kind], "marker.line.color": t.surface, "textfont.color": t.ink,
      }};
      if (kind === "connector") update = {{"line.color": t.axis}};
      // A tick is drawn with the marker's outline, so the outline carries its colour.
      if (kind === "blindtick") update = {{"marker.color": t.blind, "marker.line.color": t.blind}};
      // Both loss lines are the adapter's, told apart by their dash.
      if (kind === "loss") {{
        update = {{
          "line.color": t.adapter, "marker.color": t.adapter, "marker.line.color": t.surface,
        }};
      }}
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


def retrain_section(overall: pd.DataFrame) -> str:
    """The retraining section, or nothing when its tables have not been written."""
    paths = {
        name: TABLES_DIR / f"{name}.csv"
        for name in ("retrain", "retrain_epochs", "retrain_validation_reference")
    }
    if not all(path.exists() for path in paths.values()):
        return ""
    retrain, epochs, reference = (pd.read_csv(path) for path in paths.values())
    sections = []
    for model, spec in RETRAINED.items():
        rows = retrain_rows(retrain, overall, model)
        epochs_here = epochs[epochs["model"] == model]
        if rows.empty or epochs_here.empty:
            continue
        name = MODEL_PAIRS[model][1]
        order = list(dict.fromkeys(rows["label"]))
        chart = chart_block(
            retrain_chart(rows, order, x_range(rows["base"], rows["adapter"])),
            f"retrain-{model}", retrain_table(rows, order), None, True,
        )  # fmt: skip
        validation = reference[reference["split"] == "val"].set_index("predictor")
        released_loss = spec["measured_validation_loss"][RELEASED]
        lines = {
            "Knowing each question's average answer": validation.loc["question average", "loss"],
            "Released adapter on these questions": released_loss,
            "Lowest possible": validation.loc["lowest possible", "loss"],
        }
        loss = chart_block(
            loss_chart(epochs_here, lines), f"retrain-loss-{model}",
            epochs_table(epochs_here), None, False,
        )  # fmt: skip
        compared = retrain[
            (retrain["model"] == model)
            & (retrain["subset"] == "all items")
            & (retrain["reference"] == RELEASED)
        ]
        best_epoch = int(epochs_here.loc[epochs_here["lowest_val_loss"], "epoch"].iloc[0])
        between = retrain[
            (retrain["model"] == model)
            & (retrain["subset"] == "all items")
            & (retrain["reference"] != RELEASED)
            & (retrain["condition"] == "subpop_exact")
        ]
        sections.append(
            RETRAIN_SECTION.format(
                name=html.escape(name),
                n_epochs=int(epochs_here["epoch"].max()),
                best_epoch=best_epoch,
                max_gap=f"{math.ceil(compared['difference'].abs().max() * 10) / 10:.1f}",
                closer_low=f"{math.floor(compared['share_closer_to_reference'].min() * 100)}%",
                closer_high=f"{math.floor(compared['share_closer_to_reference'].max() * 100)}%",
                held_out=f"{int(validation['n_examples'].iloc[0]):,} prompts",
                exact_gain=f"{between['difference'].iloc[0]:.1f}",
                released_loss=f"{released_loss:.3f}",
                question_average=f"{validation.loc['question average', 'loss']:.3f}",
                chart=chart,
                loss=loss,
            )
        )
    return "\n".join(sections)


def main() -> None:
    overall = pd.read_csv(TABLES_DIR / "overall.csv")
    subsets = pd.read_csv(TABLES_DIR / "by_subset.csv")
    datasets = pd.read_csv(TABLES_DIR / "by_dataset.csv")
    blind = pd.read_csv(TABLES_DIR / "question_blind.csv")
    controls = pd.read_csv(TABLES_DIR / "calibration_controls.csv")
    blind_all = blind[blind["subset"] == "all items"].set_index("baseline")["score"]

    both = overall[overall["condition"].isin(FORMATS)]
    head_range = x_range(both["base"], both["adapter"], [p[1] for p in PUBLISHED])
    corrected_gain = (
        controls["adapter_letter_usage_matched"] - controls["base_letter_usage_matched"]
    )
    recal_range = x_range(
        controls["base_letter_usage_matched"], controls["adapter_letter_usage_matched"], [0.0]
    )

    subsets = subsets[subsets["subset"].isin(SUBSET_LABELS) & subsets["condition"].isin(FORMATS)]
    subsets = subsets.assign(label=subsets["subset"].map(SUBSET_LABELS))
    subset_order = [v for k, v in SUBSET_LABELS.items() if k in set(subsets["subset"])]
    subset_range = x_range(subsets["base"], subsets["adapter"], subsets["question_blind"])

    datasets = datasets[datasets["condition"].isin(FORMATS)].copy()
    split_names = {"pop": "Pop", "grouped": "Grouped"}
    datasets["label"] = datasets["dataset_name"] + " (" + datasets["split"].map(split_names) + ")"
    # One fixed order for both formats, so switching format never reorders the rows.
    mean_gain = datasets.groupby(["split", "label"])["gain"].mean().reset_index()
    mean_gain["split_rank"] = mean_gain["split"].map({"pop": 0, "grouped": 1})
    dataset_order = list(
        mean_gain.sort_values(["split_rank", "gain"], ascending=[True, False])["label"]
    )
    dataset_range = x_range(datasets["base"], datasets["adapter"], datasets["question_blind"])

    blocks = {"headline": [], "recalibrated": [], "subsets": [], "datasets": []}
    for condition in FORMATS:
        fig, table = headline_chart(overall, blind, condition, head_range)
        columns = {
            "label": "Model or reference",
            "base": "Base",
            "adapter": "With adapter",
            "gain": "Gain",
            "reference": "Reference score",
        }
        blocks["headline"].append(
            chart_block(fig, f"headline-{condition}", table_html(table, columns), condition, False)
        )
        fig, table = recalibrated_chart(controls, overall, condition, recal_range)
        columns = {
            "label": "Model",
            "raw_base": "Base, as run",
            "base": "Base, corrected",
            "raw_adapter": "Adapter, as run",
            "adapter": "Adapter, corrected",
            "gain": "Difference after correction",
            "reference": "Reference score",
        }
        blocks["recalibrated"].append(
            chart_block(
                fig, f"recalibrated-{condition}", table_html(table, columns), condition, False
            )
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

    # MoralMachine's human answers look reversed in SimBench's data: what reversing them does.
    flipped = pd.read_csv(TABLES_DIR / "moralmachine_reversed.csv")
    flipped = flipped[flipped["condition"].isin(FORMATS)]
    everything = flipped[flipped["subset"] == "all items"]
    mm_shift = max(
        (everything[f"{side}_reversed"] - everything[side]).abs().max()
        for side in ("base", "adapter")
    )
    mm_best = flipped.loc[flipped["subset"] == "MoralMachine", "adapter_reversed"].max()
    animals = pd.read_csv(TABLES_DIR / "moralmachine_animals_vs_people.csv")

    retrain_html = retrain_section(overall)

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
        blind_dataset=f"{blind_all['dataset']:.1f}",
        blind_benchmark=f"{blind_all['benchmark']:.1f}",
        recal_low=f"{corrected_gain.min():.0f}",
        recal_high=f"{corrected_gain.max():.0f}",
        headline="\n".join(blocks["headline"]),
        recalibrated="\n".join(blocks["recalibrated"]),
        subsets="\n".join(blocks["subsets"]),
        datasets="\n".join(blocks["datasets"]),
        retrain=retrain_html,
        exact=html.escape(exact_text),
        mm_animals=len(animals),
        mm_killed=int((animals["share_killing_people"] > 0.5).sum()),
        mm_shift=f"{mm_shift:.1f}",
        mm_best=math.ceil(mm_best),
        themes=json.dumps(THEMES),
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "subpop_on_simbench.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
