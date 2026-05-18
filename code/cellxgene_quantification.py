import cellxgene_census
import tiledbsoma
import spatialdata as sd
import spatialdata_plot

import warnings
warnings.filterwarnings("ignore")

with cellxgene_census.open_soma(census_version="2025-01-30") as census:
    #Print Census level metadata
    # print("Census level metadata\n")
    # print(list(census["census_info"].keys()))

    # print("\n")

    print("Census datasets info\n")
    print(census["census_info"]["datasets"].read().concat().to_pandas().head())
    census['census_info']['datasets'].read().concat().to_pandas().to_csv('/home/jwinchel/Syndri/code/cellxgene.csv')

    # print("\n")

    # print("Census spatial keys\n")
    # print(list(census["census_spatial_sequencing"].keys()))

    #[optional] not run - spatial data obs metadata table
    #print(census["census_spatial_sequencing"]["homo_sapiens"]["obs"].read().concat().to_pandas().head())

    #[optional] not run - spatial data var metadata table
    #print(census["census_spatial_sequencing"]["homo_sapiens"]["ms"]["RNA"]["var"].read().concat().to_pandas().head())