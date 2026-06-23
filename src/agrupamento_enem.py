from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

PRESENCE_COLS = [
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
]

ID_COLS = [
    "CO_ESCOLA",
    "NO_MUNICIPIO_ESC",
    "SG_UF_ESC",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC",
]

CLUSTER_FEATURES = SCORE_COLS + ["NOTA_MEDIA"]
DEPENDENCIA_LABELS = {
    1: "Federal",
    2: "Estadual",
    3: "Municipal",
    4: "Privada",
}
LOCALIZACAO_LABELS = {
    1: "Urbana",
    2: "Rural",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def aggregate_school_scores(
    csv_path: Path,
    chunksize: int = 300_000,
) -> tuple[pd.DataFrame, int, int]:
    """Aggregate ENEM records by school using only present students with full scores."""
    usecols = ID_COLS + PRESENCE_COLS + SCORE_COLS
    grouped_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    total_rows = 0
    valid_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin1",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk.dropna(subset=["CO_ESCOLA"]).copy()
        chunk["CO_ESCOLA"] = chunk["CO_ESCOLA"].astype("int64")

        for col in PRESENCE_COLS + SCORE_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        present_mask = chunk[PRESENCE_COLS].eq(1).all(axis=1)
        complete_score_mask = chunk[SCORE_COLS].notna().all(axis=1)
        chunk = chunk.loc[present_mask & complete_score_mask].copy()
        valid_rows += len(chunk)

        if chunk.empty:
            continue

        chunk["NOTA_MEDIA"] = chunk[SCORE_COLS].mean(axis=1)
        grouped = chunk.groupby("CO_ESCOLA")[CLUSTER_FEATURES].agg(["sum", "count"])
        grouped_parts.append(grouped)

        metadata = chunk[ID_COLS].drop_duplicates("CO_ESCOLA")
        metadata_parts.append(metadata)

    if not grouped_parts:
        raise ValueError("No valid school records were found in the ENEM file.")

    grouped_all = pd.concat(grouped_parts)
    totals = grouped_all.groupby(level=0).sum()
    school_scores = pd.DataFrame(index=totals.index)

    for col in CLUSTER_FEATURES:
        school_scores[col] = totals[(col, "sum")] / totals[(col, "count")]

    school_scores["N_PARTICIPANTES"] = totals[(SCORE_COLS[0], "count")].astype(int)

    metadata_all = (
        pd.concat(metadata_parts)
        .drop_duplicates("CO_ESCOLA")
        .set_index("CO_ESCOLA")
    )
    school_scores = school_scores.join(metadata_all, how="left").reset_index()
    school_scores["TP_DEPENDENCIA_ADM_ESC"] = pd.to_numeric(
        school_scores["TP_DEPENDENCIA_ADM_ESC"], errors="coerce"
    ).astype("Int64")
    school_scores["TP_LOCALIZACAO_ESC"] = pd.to_numeric(
        school_scores["TP_LOCALIZACAO_ESC"], errors="coerce"
    ).astype("Int64")
    school_scores["DEPENDENCIA"] = school_scores["TP_DEPENDENCIA_ADM_ESC"].map(DEPENDENCIA_LABELS)
    school_scores["LOCALIZACAO"] = school_scores["TP_LOCALIZACAO_ESC"].map(LOCALIZACAO_LABELS)

    return school_scores, total_rows, valid_rows


def evaluate_k_values(
    x_scaled: np.ndarray,
    k_values: range = range(2, 9),
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []

    for k in k_values:
        labels, centroids, inertia = kmeans(x_scaled, k, random_state=random_state, n_init=20)
        silhouette = sampled_silhouette(x_scaled, labels, random_state=random_state)
        rows.append(
            {
                "k": k,
                "inercia": inertia,
                "silhueta": silhouette,
            }
        )

    return pd.DataFrame(rows)


def choose_k(validation: pd.DataFrame) -> int:
    """Use silhouette as the primary objective rule for this short assignment."""
    return int(validation.sort_values(["silhueta", "k"], ascending=[False, True]).iloc[0]["k"])


def standardize(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    stds[stds == 0] = 1
    return (values - means) / stds


def kmeans(
    x: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        centroid_idx = rng.choice(len(x), size=n_clusters, replace=False)
        centroids = x[centroid_idx].copy()

        for _ in range(max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = centroids.copy()

            for cluster_id in range(n_clusters):
                cluster_points = x[labels == cluster_id]
                if len(cluster_points):
                    new_centroids[cluster_id] = cluster_points.mean(axis=0)
                else:
                    new_centroids[cluster_id] = x[rng.integers(0, len(x))]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        inertia = float(distances[np.arange(len(x)), labels].sum())

        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            best_inertia = inertia

    if best_labels is None or best_centroids is None:
        raise RuntimeError("K-means failed to generate a solution.")
    return best_labels, best_centroids, best_inertia


def sampled_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    sample_size: int = 2_000,
) -> float:
    rng = np.random.default_rng(random_state)
    if len(x) > sample_size:
        idx = rng.choice(len(x), size=sample_size, replace=False)
        x = x[idx]
        labels = labels[idx]

    distances = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i, label in enumerate(labels):
        same = labels == label
        other_labels = [other for other in np.unique(labels) if other != label]

        if same.sum() <= 1 or not other_labels:
            scores.append(0.0)
            continue

        a = distances[i, same].sum() / (same.sum() - 1)
        b = min(distances[i, labels == other].mean() for other in other_labels)
        scores.append((b - a) / max(a, b))

    return float(np.mean(scores))


def pca_two_components(x_scaled: np.ndarray) -> np.ndarray:
    centered = x_scaled - x_scaled.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * s[:2]


def fit_clusters(
    school_scores: pd.DataFrame,
    min_participants: int = 30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = school_scores.loc[school_scores["N_PARTICIPANTES"] >= min_participants].copy()
    model_df = model_df.dropna(subset=CLUSTER_FEATURES).copy()

    x_scaled = standardize(model_df[CLUSTER_FEATURES].to_numpy(dtype=float))

    validation = evaluate_k_values(x_scaled, random_state=random_state)
    best_k = choose_k(validation)

    labels, _, _ = kmeans(x_scaled, best_k, random_state=random_state, n_init=50)
    model_df["CLUSTER"] = labels + 1

    coords = pca_two_components(x_scaled)
    model_df["CP1_VISUAL"] = coords[:, 0]
    model_df["CP2_VISUAL"] = coords[:, 1]

    profiles = (
        model_df.groupby("CLUSTER")
        .agg(
            N_ESCOLAS=("CO_ESCOLA", "count"),
            PARTICIPANTES_MEDIOS=("N_PARTICIPANTES", "mean"),
            NOTA_MEDIA=("NOTA_MEDIA", "mean"),
            NU_NOTA_CN=("NU_NOTA_CN", "mean"),
            NU_NOTA_CH=("NU_NOTA_CH", "mean"),
            NU_NOTA_LC=("NU_NOTA_LC", "mean"),
            NU_NOTA_MT=("NU_NOTA_MT", "mean"),
            NU_NOTA_REDACAO=("NU_NOTA_REDACAO", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("NOTA_MEDIA")
    )
    ordered_clusters = {
        cluster: order + 1
        for order, cluster in enumerate(profiles["CLUSTER"].tolist())
    }
    model_df["PERFIL"] = model_df["CLUSTER"].map(ordered_clusters)
    profiles["PERFIL"] = profiles["CLUSTER"].map(ordered_clusters)
    profiles = profiles.sort_values("PERFIL").drop(columns=["CLUSTER"])

    profile_names = {
        1: "Desempenho mais baixo",
        2: "Desempenho intermediario",
        3: "Desempenho mais alto",
    }
    model_df["NOME_PERFIL"] = model_df["PERFIL"].map(profile_names).fillna(
        "Perfil " + model_df["PERFIL"].astype(str)
    )
    profiles["NOME_PERFIL"] = profiles["PERFIL"].map(profile_names).fillna(
        "Perfil " + profiles["PERFIL"].astype(str)
    )

    return model_df, validation, profiles


def save_outputs(
    clustered: pd.DataFrame,
    validation: pd.DataFrame,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    clustered.to_csv(tables_dir / "agrupamento_escolas_clusters.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(tables_dir / "agrupamento_validacao_k.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(tables_dir / "agrupamento_perfis_clusters.csv", index=False, encoding="utf-8-sig")
    write_validation_svg(validation, figures_dir / "agrupamento_cotovelo_silhueta.svg")
    write_scatter_svg(clustered, figures_dir / "agrupamento_clusters_pca.svg")
    write_profiles_svg(profiles, figures_dir / "agrupamento_perfis_notas.svg")


def scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="#222">{text}</text>'


def write_validation_svg(validation: pd.DataFrame, path: Path) -> None:
    width, height = 960, 360
    panels = [("inercia", "Metodo do cotovelo"), ("silhueta", "Indice de silhueta")]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')

    for panel_idx, (column, title) in enumerate(panels):
        x0 = 70 + panel_idx * 470
        y0, w, h = 70, 360, 220
        xs = validation["k"].to_numpy(dtype=float)
        ys = validation[column].to_numpy(dtype=float)
        x_points = [scale(x, xs.min(), xs.max(), x0, x0 + w) for x in xs]
        y_points = [scale(y, ys.min(), ys.max(), y0 + h, y0) for y in ys]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_points, y_points))
        parts.append(svg_text(x0 + w / 2, 32, title, 16))
        parts.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#444"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#444"/>')
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2f6fbb" stroke-width="3"/>')
        for k, x, y, value in zip(xs, x_points, y_points, ys):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2f6fbb"/>')
            parts.append(svg_text(x, y0 + h + 22, str(int(k)), 11))
            parts.append(svg_text(x, y - 9, f"{value:.2f}" if column == "silhueta" else f"{value/1000:.0f}k", 10))
        parts.append(svg_text(x0 + w / 2, y0 + h + 48, "Numero de grupos (k)", 12))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter_svg(clustered: pd.DataFrame, path: Path) -> None:
    width, height = 820, 560
    margin = 70
    colors = {1: "#b94e48", 2: "#d89a2b", 3: "#2f7d5f", 4: "#386cb0", 5: "#7a5195"}
    x_min, x_max = clustered["CP1_VISUAL"].min(), clustered["CP1_VISUAL"].max()
    y_min, y_max = clustered["CP2_VISUAL"].min(), clustered["CP2_VISUAL"].max()
    n_min, n_max = clustered["N_PARTICIPANTES"].min(), clustered["N_PARTICIPANTES"].max()

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(svg_text(width / 2, 32, "Escolas agrupadas por perfil de notas", 17))
    parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444"/>')

    for _, row in clustered.sample(min(len(clustered), 3500), random_state=42).iterrows():
        x = scale(row["CP1_VISUAL"], x_min, x_max, margin, width - margin)
        y = scale(row["CP2_VISUAL"], y_min, y_max, height - margin, margin)
        radius = scale(row["N_PARTICIPANTES"], n_min, n_max, 2.0, 7.0)
        color = colors.get(int(row["PERFIL"]), "#555")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.62"/>')

    legend_y = 72
    for perfil in sorted(clustered["PERFIL"].unique()):
        label = clustered.loc[clustered["PERFIL"] == perfil, "NOME_PERFIL"].iloc[0]
        color = colors.get(int(perfil), "#555")
        parts.append(f'<circle cx="{width-205}" cy="{legend_y}" r="6" fill="{color}"/>')
        parts.append(svg_text(width - 190, legend_y + 4, label, 12, anchor="start"))
        legend_y += 22

    parts.append(svg_text(width / 2, height - 22, "Componente visual 1", 12))
    parts.append(svg_text(18, height / 2, "Componente visual 2", 12, anchor="middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_profiles_svg(profiles: pd.DataFrame, path: Path) -> None:
    width, height = 980, 460
    margin_left, margin_bottom, margin_top = 70, 80, 55
    plot_w, plot_h = 820, 300
    variables = SCORE_COLS + ["NOTA_MEDIA"]
    colors = {1: "#b94e48", 2: "#d89a2b", 3: "#2f7d5f", 4: "#386cb0", 5: "#7a5195"}
    y_max = profiles[variables].to_numpy(dtype=float).max() * 1.08
    group_w = plot_w / len(variables)
    bar_w = group_w / (len(profiles) + 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(svg_text(width / 2, 30, "Caracterizacao dos grupos pelas notas medias", 17))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#444"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#444"/>')

    for var_idx, var in enumerate(variables):
        x_base = margin_left + var_idx * group_w
        parts.append(svg_text(x_base + group_w / 2, margin_top + plot_h + 28, var.replace("NU_NOTA_", ""), 11))
        for profile_idx, (_, row) in enumerate(profiles.iterrows()):
            value = float(row[var])
            bar_h = scale(value, 0, y_max, 0, plot_h)
            x = x_base + (profile_idx + 0.5) * bar_w
            y = margin_top + plot_h - bar_h
            color = colors.get(int(row["PERFIL"]), "#555")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{bar_h:.1f}" fill="{color}"/>')

    legend_y = 70
    for _, row in profiles.iterrows():
        color = colors.get(int(row["PERFIL"]), "#555")
        parts.append(f'<rect x="{width-205}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
        parts.append(svg_text(width - 186, legend_y, row["NOME_PERFIL"], 12, anchor="start"))
        legend_y += 22

    parts.append(svg_text(35, margin_top + plot_h / 2, "Media", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_conclusion(
    total_rows: int,
    valid_rows: int,
    clustered: pd.DataFrame,
    validation: pd.DataFrame,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_k = int(validation.sort_values("silhueta", ascending=False).iloc[0]["k"])
    best_silhouette = validation.sort_values("silhueta", ascending=False).iloc[0]["silhueta"]
    low = profiles.iloc[0]
    high = profiles.iloc[-1]
    spread = high["NOTA_MEDIA"] - low["NOTA_MEDIA"]

    text = f"""Analise de agrupamento - ENEM 2024

Tecnica utilizada: K-medias com variaveis padronizadas e distancia euclidiana.
Unidade de analise: escola.
Variaveis usadas: notas medias de Ciencias da Natureza, Ciencias Humanas, Linguagens, Matematica, Redacao e media geral.

Registros lidos: {total_rows:,}
Registros validos com escola, presenca nas provas e notas completas: {valid_rows:,}
Escolas analisadas apos filtro de pelo menos 30 participantes: {len(clustered):,}

Validacao:
O numero de grupos foi avaliado de k=2 a k=8 por inercia e indice de silhueta.
A regra objetiva pela maior silhueta selecionou k={best_k}, com silhueta={best_silhouette:.4f}.

Principais resultados:
Foram identificados {best_k} perfis de escolas. O perfil de menor desempenho tem media geral de {low['NOTA_MEDIA']:.2f} pontos, enquanto o perfil de maior desempenho tem media geral de {high['NOTA_MEDIA']:.2f} pontos. A diferenca entre esses extremos e de {spread:.2f} pontos.

Interpretacao:
Os grupos se diferenciam principalmente pelo nivel medio de desempenho nas cinco areas, com Matematica e Redacao contribuindo para ampliar a distancia entre os perfis. Como as variaveis foram padronizadas, nenhuma prova domina o calculo apenas por escala. A analise deve ser lida como descricao de perfis de desempenho escolar, nao como evidencia causal.
"""
    (output_dir / "tables" / "agrupamento_conclusao.txt").write_text(text, encoding="utf-8")


def run_analysis(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    min_participants: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = project_root()
    csv_path = csv_path or root / "data" / "raw" / "RESULTADOS_2024.csv"
    output_dir = output_dir or root / "outputs"

    school_scores, total_rows, valid_rows = aggregate_school_scores(csv_path)
    clustered, validation, profiles = fit_clusters(
        school_scores,
        min_participants=min_participants,
    )
    save_outputs(clustered, validation, profiles, output_dir)
    write_conclusion(total_rows, valid_rows, clustered, validation, profiles, output_dir)
    return clustered, validation, profiles


if __name__ == "__main__":
    clustered_df, validation_df, profiles_df = run_analysis()
    print("Analise de agrupamento concluida.")
    print(validation_df.to_string(index=False))
    print(profiles_df.to_string(index=False))
