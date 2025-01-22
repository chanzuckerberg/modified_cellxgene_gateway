from cellxgene_gateway.flask_app import app
import os


# scvi-tools, CELLxGENE Census and TileDB SOMA
import cellxgene_census
from cellxgene_census.experimental import get_embedding
import tiledbsoma as soma
import scvi
from datetime import datetime

# single cell
import scanpy as sc
import anndata as ad

# scientific computing
import numpy as np
import pandas as pd
import scipy.sparse as sp
from flask import jsonify, request
import requests

species_to_common_map = {"homo_sapiens": "human", "mus_musculus": "mouse"}


@app.route("/subcell/census_versions.json", methods=["GET"])
def get_census_versions():
    versions = cellxgene_census.get_census_version_directory()
    return jsonify(list(versions.keys()))


def download_cxg_model(version, species, common_name) -> None:
    ## download the model so we can run it locally
    dest_file = f"scvi-{common_name}-{version}/model.pt"
    if os.path.exists(dest_file):
        return
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    remote_file = f"https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/{version}/{species}/model.pt"

    res = requests.get(remote_file)
    if res.status_code == 200:  # http 200 means success
        with open(dest_file, "wb") as file_handle:  # wb means Write Binary
            file_handle.write(res.content)


@app.route("/subcell/generate/<path:path>", methods=["POST"])
def run_subcell(path):
    # TODO, don't hardcode this
    #
    # Configure Global Variables
    ## Set latest Census Version
    version = request.form.get("version")
    if not version:
        version = "2024-07-01"
    species = request.form.get("species")
    if not species:
        species = "human"
    try:
        common_name = species_to_common_map[species]
    except KeyError:
        raise Exception(
            f"Invalid species, must be one of: {', '.join(species_to_common_map.keys())}"
        )

    # TODO don't hard-code the path
    path = "human_brain_single_cell.h5ad"
    path_parts = os.path.splitext(path)
    output_path = (
        path_parts[0] + "_" + datetime.now().strftime("%Y%m%d_%H%I%S") + path_parts[1]
    )

    adata = ad.read_h5ad(path)
    # Populate and rename a few metadata fields to make visualization easier downstream
    adata.obs["soma_joinid"] = list(range(adata.n_obs))
    adata.obs["tissue"] = "brain"
    adata.obs["tissue_general"] = "brain"
    adata.obs["disease"] = "normal"
    adata.obs["dataset_id"] = "jingjing"
    adata.obs["batch"] = "scvi-filler-value"

    cell_type_mappings = {
        "EN": "excitatory neuron",
        "EN_newborn": "newly differentiated excitatory neuron",
        "RG": "radial glial cell",
        "IN-CGE": "caudal ganglionic eminence derived interneuron",
        "IN-MGE": "medial ganglionic eminence derived interneuron",
        "OPC": "oligodendrocyte precursor cell",
        "EN-Non-IT": "non-intratelencephalic excitatory neuron",
        "Oligodendrocyte": "oligodendrocyte",
        "IPC-EN": "intermediate progenitor cell",
    }

    adata.obs.rename(columns={"celltype": "cell_type"}, inplace=True)
    adata.obs["cell_type"].replace(cell_type_mappings, inplace=True)

    adata.var.reset_index(inplace=True)
    adata.var.rename(
        columns={"gene_name": "gene_symbol", "geneid": "gene_id"}, inplace=True
    )
    adata.var.set_index("gene_id", inplace=True)
    adata.var.drop(columns=["tmp", "mito"], inplace=True)
    adata.var.index.name = ""

    download_cxg_model(version, species, common_name)

    # # Find the latest scVI model version available from CELLxGENE Census API
    with cellxgene_census.open_soma(census_version=version) as census:
        census = cellxgene_census.open_soma(census_version=version)
        scvi_info = cellxgene_census.experimental.get_embedding_metadata_by_name(
            embedding_name="scvi",
            organism="homo_sapiens",
            census_version=version,
        )

    scvi_info["model_link"]

    scvi_adata = scvi.model.SCVI.prepare_query_anndata(
        adata,
        f"scvi-{common_name}-{version}",
        return_reference_var_names=False,
        inplace=False,
    )  # for some reason running inplace doesn't actually modify as expected

    vae_q = scvi.model.SCVI.load_query_data(
        scvi_adata,
        "scvi-{common_name}-{version}",
    )

    # This allows for a simple forward pass
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()
    scvi_adata.obsm["scvi"] = latent

    combined_adata = ad.concat([ref_adata, scvi_adata], join="outer", label="my_data")
    #
    # Tidy up obs to have just the columns we want and deal with missing values
    obs_cols = [
        "soma_joinid",
        "tissue",
        "tissue_general",
        "cell_type",
        "disease",
        "dataset_id",
        "my_data",
    ]
    combined_adata.obs = combined_adata.obs[obs_cols]
    combined_adata.obs.replace("nan", np.nan, inplace=True)
    combined_adata.obs["my_data"] = np.where(  # human readable labels
        combined_adata.obs["my_data"] == 1, "My data", "Reference"
    )

    # Remove unused categorical labels
    for col in combined_adata.obs.select_dtypes("category").columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

    # Set the index
    combined_adata.obs["soma_joinid"] = combined_adata.obs["soma_joinid"].astype("str")
    combined_adata.obs.set_index("soma_joinid", inplace=True)
    combined_adata.obs_names = combined_adata.obs.index
    combined_adata.var["feature_name"] = ref_adata.var["feature_name"]
    combined_adata.var_names = combined_adata.var["feature_name"]

    # normalize for viz purposes
    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)

    # run UMAP on top of scVI embedding
    sc.pp.neighbors(combined_adata, use_rep="scvi")
    sc.tl.umap(combined_adata)
    combined_adata.write_h5ad(output_path)
    return jsonify({"success": True})
