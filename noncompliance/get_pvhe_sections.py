#
# Copyright 2026 Fabio Mazza
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from .lib import process_votes, SEZIONE_COL, renorm
import argparse


# ── Funzioni ────────────────────────────────────────────────────────────────

def load_data_elections(city: str,
                        year_vote: int,
                        kind: str,
                        base_path: Path|str = "."
                        ) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str,list[str]]]:
    electoral_path = Path(base_path) / "electoral_data"
    sezioni = gpd.read_parquet(electoral_path / f"sezioni_elettorali_{city}_{kind}.parq")
    sezioni["sezione"] = sezioni["SEZIONE"].astype(int)

    files_votes = {
        "milano": f"camera_liste_{year_vote}.csv",
        "palermo": f"politiche_camera_{year_vote}.csv",
        "torino": "pol2022_liste.csv" if year_vote == 2022 else f"pol{year_vote}_camera_collegi.csv",
        "bologna": f"elezioni-politiche-{year_vote}-voti-liste-camera-dei-deputati.csv",
        "firenze": f"camera_{year_vote}.csv",
        "roma": f"camera_{year_vote}.csv"
    }
    voti_df = pd.read_csv(electoral_path / "results" / city / files_votes[city])

    with open(electoral_path/"Parties_map.json", "rt") as f:
        map_code_party = json.load(f)

    return sezioni, voti_df, map_code_party


def compute_voti_agg(voti_df: pd.DataFrame, sezioni: gpd.GeoDataFrame,
                     map_code_party: dict[str,list[str]], city: str, year_vote: int) -> pd.DataFrame:
    cleanup_columns = (city == "torino" and year_vote == 2018)
    remove_voti_lista = cleanup_columns
    return process_votes(
        voti_df, sezioni, map_code_party,
        cleanup_columns=cleanup_columns,
        remove_voti_lista_name=remove_voti_lista,
    )


def compute_vhe(voti_agg: pd.DataFrame, coeffs_vhe: pd.DataFrame,
                period: int) -> tuple[pd.DataFrame, float]:
    """Calcola il VHE medio per sezione e la media città."""
    coeffs = coeffs_vhe[["party_name_short", "party_name", "OLS_coefficient"]]

    # Media città
    u = (voti_agg.groupby("partito")["voti"].sum() / len(voti_agg[SEZIONE_COL].unique())).reset_index()
    u = u.merge(coeffs, how="left", left_on="partito", right_on="party_name_short")
    mean_vhe_city = (u["voti"] * u["OLS_coefficient"]).sum()

    # VHE per sezione
    df = voti_agg.merge(coeffs, how="left", left_on="partito", right_on="party_name_short")
    result = (
        df.assign(mean_VHE=df["voti"] * df["OLS_coefficient"])
        .groupby(SEZIONE_COL)["mean_VHE"]
        .sum()
        .reset_index()
    )
    result["mean_VHE_bal"] = result["mean_VHE"] - mean_vhe_city
    result["non_compliance"] = renorm(result["mean_VHE"].values)
    result.rename(columns={SEZIONE_COL: "sezione"}, inplace=True)

    return result, mean_vhe_city


def make_party_plot(sezioni: gpd.GeoDataFrame, voti_agg: pd.DataFrame, 
                    party:str="FdI", 
                    cmap:str="cividis",
                    figsize=(9,6)) -> plt.Figure:
    dfsezvot = sezioni.merge(voti_agg[voti_agg["partito"] == party], right_on="SEZIONE", left_on="sezione")
    colv = dfsezvot["voti"] / dfsezvot["voti"].max()
    cmap = plt.get_cmap(cmap)

    f, ax = plt.subplots(figsize=figsize)
    dfsezvot.plot(ax=ax, color=cmap(colv), legend=True, legend_kwds={"label": ""})
    sezioni.boundary.plot(ax=ax, lw=0.2, color="black")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=dfsezvot["voti"].min(), vmax=dfsezvot["voti"].max()
    ))
    cbr = f.colorbar(sm, ax=ax)
    cbr.set_label(f"Fraction of votes for {party}", fontsize=11)
    plt.subplots_adjust()
    return f


def make_vhe_map(sezioni: gpd.GeoDataFrame, result: pd.DataFrame) -> plt.Figure:
    sec_mvhe = sezioni.merge(result, on="sezione")
    f, ax = plt.subplots(figsize=(10, 8))
    sec_mvhe.plot("mean_VHE", cmap="viridis", ax=ax, legend=True)
    ax.axis("off")
    return f


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("city", ) #choices=["torino", "milano", "palermo"])
    parser.add_argument("-y", "--year", type=int, choices=[2022, 2018], default=2022)
    parser.add_argument("--kind", default="census", help="kind of electoral precincts grouping")
    parser.add_argument("-p","--period", default=3, type=int, help="period of VHE fit")
    return parser.parse_args()


def main():
    args = parse_args()
    city = args.city
    year_vote = args.year
    kind = args.kind
    period = args.period

    # Caricamento dati
    sezioni, voti_df, map_code_party = load_data_elections(city, year_vote, kind)
    print("Colonne voti:", voti_df.columns.tolist())

    # Calcoli
    voti_agg = compute_voti_agg(voti_df, sezioni, map_code_party, city, year_vote)
    coeffs_vhe = pd.read_csv(f"paoletti2024/coefficients_ita_save_per_{period}.csv")
    result, mean_vhe_city = compute_vhe(voti_agg, coeffs_vhe, period)
    print("MEAN CITY:", mean_vhe_city)

    # Salvataggio dati
    out_folder = Path("data_generated")
    out_folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        out_folder / f"{city}_elect_{year_vote}_vhe_mean_sec_{kind}_per_{period}.csv",
        index=False,
    )

    # Salvataggio figure
    out_images = Path("images")
    out_images.mkdir(parents=True, exist_ok=True)

    party_plot="PD"
    fig_fdi = make_party_plot(sezioni, voti_agg,party_plot)
    fig_fdi.savefig(
        out_images/f"sezioni_{city}_{kind}_election_{year_vote}_votes_{party_plot}.png",
        bbox_inches="tight", dpi=200,
    )
    plt.close(fig_fdi)

    fig_vhe = make_vhe_map(sezioni, result)
    fig_vhe.savefig(
        out_images / f"sezioni_{city}_{kind}_election_{year_vote}_pvhe_per_{period}.png",
        bbox_inches="tight", dpi=200,
    )
    plt.close(fig_vhe)


if __name__ == "__main__":
    main()