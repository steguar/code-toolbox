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

from matplotlib.axes import Axes

from typing import Dict,Union
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

SEZIONE_COL="SEZIONE"
VOTI_VALIDI_COL="TOTALE"

CRS_WEBM = "EPSG:3857"
def renorm(x: Union[list[float],np.ndarray,pd.Series]):
    M = np.max(x)
    m = np.min(x)
    r = (x-m)/(M-m)
    return r

def renorm_nan(x: Union[list[float],np.ndarray,pd.Series]):
    M = np.nanmax(x)
    m = np.nanmin(x)
    r = (x-m)/(M-m)
    return r

def mkdir_if_needed(p):
    if not p.exists():
        p.mkdir(parents=True)

def plot_heatmap(df: pd.DataFrame, x:str, y:str, z:str,ax:Axes,cmap="viridis", **kwargs):  # pyright: ignore[reportMissingParameterType]
    nnp=df.pivot(columns=x, index=y,values=z)
    ## reverse order of rows (y axis) because the top has higher values
    out = sns.heatmap(nnp[::-1], cmap=cmap, ax=ax,**kwargs)
    ax.axis("equal")
    return out

def process_votes(voti_df: pd.DataFrame,sezioni: pd.DataFrame, map_code_parties: Dict[str,list[str]], cleanup_columns:bool=False, remove_voti_lista_name:bool=False):
    if len(voti_df[voti_df.TOTALE==0]) > 0:
        msg = "Have {} rows that have 0 as total votes: {}".format(sum(voti_df.TOTALE == 0), voti_df["SEZIONE"][voti_df.TOTALE==0].values)
        #raise (msg)
        print("WARNING: ", msg)
    sez_strange = voti_df[np.isnan(voti_df["TOTALE"])]["SEZIONE"]
    assert len(sez_strange) == 0

    voti_df_filt = voti_df[voti_df["SEZIONE"].isin(sezioni["sezione"])]
    if cleanup_columns: 
        cols = voti_df_filt.columns
        cols_uni = list(filter(lambda x: "UNINOMINALE" in x, cols))
        print("Removing columns:",cols_uni)
        if len(cols_uni)>0:
            voti_df_filt.drop(cols_uni, axis=1, inplace=True)

    if remove_voti_lista_name:
        ##remove names from "VOTI ALLA LISTA"
        cols = list(voti_df_filt.columns)
        nc = []
        for c in cols:
            c= str(c)
            if "VOTI ALLA LISTA" in c:
                u=c.removesuffix("- VOTI ALLA LISTA").strip()
            else: u=c

            nc.append(u)
        voti_df_filt.columns = nc

    colonne_keep = ["SEZIONE","MUNICIPIO","TOTALE",]
    colonne_keep = set(colonne_keep).intersection(voti_df_filt.columns)
    colonne_keep = list(colonne_keep)

    print("Colonne da tenere:", colonne_keep)

    COLS_EXCLUDE = ["AFFLUENZA", "TOTALE_ISCRITTI"]

    COLS_INCLUDE = [SEZIONE_COL, VOTI_VALIDI_COL,]

    cols_excl = list( set(COLS_EXCLUDE).intersection(voti_df_filt.columns) )
    print("REMOVING COLS ", cols_excl)
    voti_df_melt = voti_df_filt.drop(cols_excl, axis=1) 
    cols_partiti = set(voti_df_melt.columns).difference(COLS_INCLUDE)

    rename_map = {c: c.upper() for c in cols_partiti if c != c.upper()}
    if rename_map:
        voti_df_melt = voti_df_melt.rename(columns=rename_map)
        cols_partiti = {c.upper() for c in cols_partiti}

    #MAP_COLS_PARTIES = {l:k for k,v in map_code_parties.items() for l in v}
    MAP_COLS_PARTIES = {l.upper(): k for k, v in map_code_parties.items() for l in v}

    #assert np.all([c in MAP_COLS_PARTIES.keys() for c in cols_partiti])
    for c in cols_partiti:
        if c not in MAP_COLS_PARTIES.keys():
            msg = f"Cannot assign column `{c}` to any party"
            raise ValueError(msg)

    voti_df_tomelt = voti_df_melt.copy()
    vv = list(set(voti_df_melt.columns).difference(COLS_INCLUDE))
    for k in vv:
        print(k)
        voti_df_tomelt[f"{k}_frac"] = voti_df_tomelt.loc[:,k] / voti_df_tomelt[VOTI_VALIDI_COL]
    voti_df_tomelt = voti_df_tomelt.drop(vv, axis=1)

    #replacemap= {f"{k}": v for k,v in MAP_COLS_PARTIES.items()}
    #dft1=pd.melt(voti_df_melt, id_vars=COLS_INCLUDE, var_name="partito", value_name="voti").replace(replacemap)
    #dft1.groupby(["SEZIONE"]+["partito"]).agg({"voti": "sum"}).reset_index()
    #votes_sec: DataFrame = dft1.groupby(COLS_INCLUDE+["partito"]).agg({"voti": "sum"}).reset_index()

    voti_melted = pd.melt(voti_df_tomelt, id_vars=COLS_INCLUDE,var_name="partito",value_name="voti" )
    PARTITO_REPLACE= {f"{k}_frac": v for k,v in MAP_COLS_PARTIES.items()}

    voti_agg = voti_melted.replace(PARTITO_REPLACE)

    voti_agg = voti_agg.groupby(COLS_INCLUDE+["partito"]).agg({"voti": "sum"}).reset_index()
    return voti_agg

def project_pvhe_tiles(tiles: gpd.GeoDataFrame, sections: gpd.GeoDataFrame, pvhe_scores: pd.DataFrame, 
    geo_col_tiles:str="border", plot=False,
    ):
    tiles = tiles.set_geometry(geo_col_tiles).to_crs(CRS_WEBM)
    sections["sezione"] = sections["SEZIONE"].astype(int)

    sect_data = sections[["sezione","geometry"]].to_crs(CRS_WEBM).sort_values("sezione")
    sect_data.reset_index(drop=True,inplace=True)
    u=[
    tiles[geo_col_tiles].intersection(sect_data.geometry[i]).area for i in range(len(sect_data))]

    areas=np.stack(u)

    comp_mat=areas / areas.sum(0)

    err_tid = np.where(areas.sum(0)==0)[0]
    if len(err_tid)>0:
        print(f"PROBLEM: Tiles {err_tid} have null sum of area, cannot continue")

    s = sect_data.area

    si = np.round(s).astype(int)

    bins = np.logspace(3, 7, 81)
    bins
    if plot:
        cs,_ = np.histogram(s, bins)
        fig = plt.figure()
        plt.bar(bins[:-1], cs,width=(bins[1:]-bins[:-1]), align="edge", label="Areas of sections")
        plt.xscale("log")
        plt.vlines(np.mean(tiles.area),*plt.ylim(), color="red", label="area of tile")
        #plt.hist(tiles.area, bins=30, label="Areas of tiles", color="red", zorder=10)

        plt.xlim((1e3,None))
        plt.legend()
    else:
        fig = None
    
    sections_vhe = sect_data.merge(pvhe_scores, on="sezione")
    tiles_pvhe=comp_mat.T.dot(pvhe_scores["mean_VHE"])
    #if normalize:
    noncompl = renorm_nan(tiles_pvhe)
    #else:
    #    noncompl = tiles_pvhe
    tiles["pvhe"] = tiles_pvhe
    tiles["non_compliance"] = noncompl

    return tiles, sections_vhe, fig