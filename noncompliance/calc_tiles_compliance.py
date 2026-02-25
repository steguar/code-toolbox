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
import contextily as cx

from .lib import project_pvhe_tiles, CRS_WEBM, mkdir_if_needed, plot_heatmap
from typing import Union
import argparse

from .get_pvhe_sections import load_data_elections, compute_voti_agg, compute_vhe, make_vhe_map

parser = argparse.ArgumentParser()
parser.add_argument("city", ) #choices=["torino","milano","palermo"])
parser.add_argument("-y","--year", type=int, choices=[2022,2018], default=2022)
parser.add_argument("-p","--period", type=int, default=3, help="VHE fit period")
parser.add_argument("--kind", default="census", help="kind of electoral precincts grouping")

#parser.add_argument("--tile_size", type=int, default=250, help="tile size in meters")
parser.add_argument("--ec","--extra_city_name", dest="extra_city_name",default="250m")
parser.add_argument("--usnfold","--usn_pop_folder", dest="usn_folder", default="../../usn_cleaned/data/population", help="USN Population folder")
parser.add_argument("--save_debug",action="store_true")

args = parser.parse_args()

POP_FOLDER = Path(args.usn_folder)
SEC_TYPE = "census"
CITY = args.city
ANNO = args.year
PERIOD= args.period
kind = args.kind
fold_city_usn = args.city
if len(args.extra_city_name) > 0:
    fold_city_usn += f"_{args.extra_city_name}"
tiles_pop_city =gpd.read_parquet(POP_FOLDER / fold_city_usn / "tiles_data_pop.parq")
 #f"../usn_cleaned/data/population/{city_folder}/tiles_data_pop.parq"

tiles = tiles_pop_city.drop(columns=["corner","center"]).set_geometry("border")#.to_crs(CRS_WEBM)

tiles = tiles.rename(columns={"count":"people"}).drop(columns=["i","j"])

sezioni, voti_df, map_code_party = load_data_elections(CITY, ANNO, kind)
print("Colonne voti:", voti_df.columns.tolist())

OUT_FOLDER = Path("data_generated")
mkdir_if_needed(OUT_FOLDER)

# Calcoli
voti_agg = compute_voti_agg(voti_df, sezioni, map_code_party, CITY, ANNO)
coeffs_vhe = pd.read_csv(f"paoletti2024/coefficients_ita_save_per_{PERIOD}.csv")
pvhe_scores_secs, mean_vhe_city = compute_vhe(voti_agg, coeffs_vhe, PERIOD)
print("MEAN CITY:", mean_vhe_city)

pvhe_scores_secs.to_csv(
    OUT_FOLDER / f"{CITY}_elect_{ANNO}_vhe_mean_sec_{kind}_per_{PERIOD}.csv",
    index=False,)

tiles_pvhe, sections_pvhe, figure = project_pvhe_tiles(tiles, sezioni, pvhe_scores_secs )
tiles_ex = tiles_pvhe[np.isnan(tiles_pvhe.pvhe)]
tiles_good = tiles_pvhe[~np.isnan(tiles_pvhe.pvhe)]



Fold_images = Path("images") 
mkdir_if_needed(Fold_images)

fig_vhe = make_vhe_map(sezioni, pvhe_scores_secs)
fig_vhe.savefig(
    Fold_images / f"sezioni_{CITY}_{kind}_election_{ANNO}_pvhe_per_{PERIOD}.png",
    bbox_inches="tight", dpi=200,
)
plt.close(fig_vhe)

fold_save = OUT_FOLDER / fold_city_usn
fold_save.mkdir(parents=True, exist_ok=True)
tiles_good.drop(columns=["border"]).to_csv(fold_save/f"tiles_data_vaxhes_{ANNO}_period_{PERIOD}.csv",index=False)
if len(tiles_ex)>0:
    tiles_ex.drop(columns=["border"]).to_csv(fold_save/f"tiles_data_vaxhes_{ANNO}_period_{PERIOD}_excluded.csv",index=False)
    print("Tiles to exclude saved")
base_name = f"{CITY}_{ANNO}_period_{PERIOD}_prectype_{SEC_TYPE}"

#figure.savefig(Fold_images / "")
if args.save_debug:
    figure.savefig(Fold_images /f"{base_name}_tiles_sect_comparison.png", bbox_inches="tight", dpi=150)
plt.close(figure)

f,ax = plt.subplots(1,2,figsize=(12,6))
plot_heatmap(tiles_pvhe, "x","y","pvhe",ax[0], cbar=False)
sections_pvhe.plot("mean_VHE",ax=ax[1])
ax[1].axis("off")
plt.savefig(Fold_images/f"{base_name}_tiles_sect_pvhe.png", bbox_inches="tight", dpi=200)
plt.close()

ax=tiles_good.plot("non_compliance", figsize=(9,6), alpha=0.9, legend=True, 
    cmap="viridis", legend_kwds={"label":"Non-compliance"})
cx.add_basemap(ax, crs=tiles_good.crs,  source=cx.providers.CartoDB.Voyager)
ax.axis("off")
plt.savefig(Fold_images/f"{base_name}_noncompliance.png", bbox_inches="tight", dpi=200)
plt.close()
