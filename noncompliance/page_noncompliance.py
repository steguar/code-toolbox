import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from io import BytesIO
from .get_pvhe_sections import load_data_elections, compute_voti_agg, compute_vhe, make_vhe_map, make_party_plot
from .lib import project_pvhe_tiles, CRS_WEBM, mkdir_if_needed, plot_heatmap

CITIES = ["torino","milano","palermo","roma"]
YEARS = [2018,2022]

@st.cache_data
def _load_st_elections(city: str,
                        year_vote: int,
                        kind: str,
                        base_path: Path|str = "."
                        ) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str,list[str]]]:
    return load_data_elections(city, year_vote, kind, base_path)

@st.cache_data
def load_geoparq(file):    
    return gpd.read_parquet(file)

@st.cache_data
def load_coeffs(fold:Path, period):
    return pd.read_csv(fold/f"paoletti2024/coefficients_ita_save_per_{period}.csv")

#@st.cache_data
def read_tiles_project(tiles_f,sezioni, pvhe_scores_secs):
    tiles = gpd.read_parquet(tiles_f)
    if "center" in tiles.columns:
        tiles.drop('center', axis=1, inplace=True)
    if "corner" in tiles.columns:
        tiles.drop('corner', axis=1, inplace=True)

    tiles_pvhe, sections_pvhe, figure = project_pvhe_tiles(tiles, sezioni, pvhe_scores_secs )
    return tiles_pvhe


def make_page(base_path:Path|str = "."):
    if "tiles_parquet_vals" not in st.session_state:
        st.session_state["tiles_parquet_vals"] = None

    st.header("Non-compliance calculation for cities")
    st.markdown("Here are the parameters for selecting the data. Note that not every combination is proven to work"
                " (might be because of missing or incomplete data)")
    col1, col2 = st.columns(2)


    with col1:
        CITY = st.selectbox("City", CITIES)
        PERIOD = st.selectbox("VHE coefficients period", [2,3,4], index=1)

    with col2:
        ANNO = st.selectbox("Year for electoral results", YEARS,index=1)
        kind = st.selectbox("Precincts type", ["census","original"])

    file_upload = st.file_uploader("Upload `tiles_data_pop.parq` from the population data folder in USN (e.g. usn/data/population/city)")
    
    COMMON_NAME = f"{CITY}_year_{ANNO}_vheper_{PERIOD}_precincts_{kind}"

    sezioni, voti_df, map_code_party = _load_st_elections(CITY, ANNO, kind, base_path)

    voti_agg = compute_voti_agg(voti_df, sezioni, map_code_party, CITY, ANNO)

    coeffs_vhe = load_coeffs(Path(base_path), PERIOD)
    pvhe_scores_secs, mean_vhe_city = compute_vhe(voti_agg, coeffs_vhe, PERIOD)
    sec_mvhe = sezioni.merge(pvhe_scores_secs, on="sezione")

    if file_upload is not None:
        tiles_pvhe = read_tiles_project(file_upload, sezioni, pvhe_scores_secs)
    else:
        tiles_pvhe = None
    #st.text("Below, you can plot the votes by party inside of each city precinct"
    #        "")

    st.subheader("Download results")
    with st.expander("**Plot of the votes by precinct**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            party = st.selectbox("Party", list(map_code_party.keys()))
        
        fig = make_party_plot(sezioni, voti_agg, party, figsize=(9,5))
        st.pyplot(fig)
    cols = st.columns(2)
    with cols[0]:
        buffer = BytesIO()
        sec_mvhe.to_parquet(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="Download sections noncompliance data",
            data=buffer,
            file_name=f"sections_non_compliance_{COMMON_NAME}.parquet",
            mime="application/vnd.apache.parquet",
        )
    with cols[1]:
        if tiles_pvhe is not None:
            tiles_ex = tiles_pvhe[np.isnan(tiles_pvhe.pvhe)]
            tiles_good = tiles_pvhe[~np.isnan(tiles_pvhe.pvhe)]

            if len(tiles_ex)>0:
                st.warning(f"**WARNING**: {len(tiles_ex)} tiles do not correspond to sections, see below")
                st.dataframe(tiles_ex.drop(columns=["border"]))
            if len(tiles_good)>0:
                buffer = BytesIO()
                tiles_good.drop(columns=["border"]).to_csv(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="Download tiles noncompliance (CSV)",
                    data=buffer,
                    file_name=f"tiles_non_compliance_{COMMON_NAME}.csv",
                    mime="text/csv",
                )
            else:
                st.error("No non-compliance data obtained for the files, are you sure that the combination of parameters is compatible with the tiles uploaded?")
        else:
            tiles_good = None


    st.subheader("Plots of the non-compliance")
    col1, col2 = st.columns(2)
    with col1:
        f,ax = plt.subplots(figsize=(7,5.5))
        #plot_heatmap(tiles_pvhe, "x","y","pvhe",ax, cbar=True)
        #ax[1].axis("off")
        sec_mvhe.plot("non_compliance",ax=ax,legend=True)
        ax.axis("off")

        ax.set_title("Precincts non-compliance")
        st.pyplot(f)
    with col2:
        if tiles_good is not None and len(tiles_good)>0:
            f,ax = plt.subplots(figsize=(7,5.5))
            plot_heatmap(tiles_good, "x","y","non_compliance",ax, cbar=True)
            #ax[1].axis("off")
            ax.set_title("Tiles non-compliance")
            st.pyplot(f)
        #plt.savefig(Fold_images/f"{base_name}_tiles_sect_pvhe.png", bbox_inches="tight", dpi=200)

