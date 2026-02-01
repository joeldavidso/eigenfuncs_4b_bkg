import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm


def Get_Cuts(batch):

    resolved_bool = np.logical_and(np.array(batch["dEta_hh"]) < 1.5, np.array(batch["pass_resolved"]))
    
    SR_bool = np.logical_and(np.array(batch["X_hh"]) < 1.6, resolved_bool)
    CR_bool = np.logical_and(np.array(batch["X_hh"]) > 1.6, resolved_bool)

    NJ_bools = {"2b2j": np.array(batch["ntag"]) == 2,
                "3b1j": np.array(batch["ntag"]) == 3,
                "4b": np.array(batch["ntag"]) >= 4}

    return SR_bool, CR_bool, NJ_bools


def Run_Over_Events(filepath):
    dataset = pq.ParquetFile(filepath)
    categories = ["2b2j", "3b1j", "4b"]
    
    # 1. Initialize writers for each category
    writers = {}
    for nj in categories:
        output_file = filepath[:-8] + f"_reduced_{nj}.parquet"
        if os.path.isfile(output_file):
            os.remove(output_file)
        writers[nj] = None # We initialize the writer on the first valid batch

    # 2. Process and write batch-by-batch
    for batch in tqdm(dataset.iter_batches(batch_size=50_000), desc=filepath):
        SR_bool, CR_bool, NJ_bools = Get_Cuts(batch)
        
        for nj in categories:
            resolved_mask = np.logical_or(SR_bool, CR_bool)
            mask = np.logical_and(NJ_bools[nj], resolved_mask)
            if np.any(mask): # Only process if there are events for this category
                # Create a small temporary dataframe for this batch
                df_batch = pd.DataFrame({
                    "m_h1": batch["m_h1"].to_numpy()[mask],
                    "m_h2": batch["m_h2"].to_numpy()[mask],
                    "SR": SR_bool[mask]
                })
                
                table = pa.Table.from_pandas(df_batch)
                
                # Initialize writer on the first batch with actual data
                if writers[nj] is None:
                    output_file = filepath[:-8] + f"_reduced_{nj}.parquet"
                    writers[nj] = pq.ParquetWriter(output_file, table.schema, compression='gzip')
                
                writers[nj].write_table(table)

    # 3. Clean up and close writers
    for nj in writers:
        if writers[nj] is not None:
            writers[nj].close()


def Validate_Saved_Files(filepath, NJ):

    NSR, NCR = 0,0
    NTotal = 0

    dataset = pq.ParquetFile(filepath[:-8] + "_reduced_"+NJ+".parquet")

    for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size=10_000), desc=filepath)):


        NTotal += np.array(batch["SR"]).shape[0]

        NCR += np.sum(np.logical_not(np.array(batch["SR"])))
        NSR += np.sum(np.array(batch["SR"]) )

    print("N_events: " + str(NTotal))
    print("N_CR: " + str(NCR))
    print("N_SR: " + str(NSR))

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


# This Code will read in 2b/3b/4b parquet file data and output
# a parquet file with the mH1 and mH2 values saved alongside
# a CR/SR boolean variable, split into 2b, 3b, 4b files
if __name__ == "__main__":

    sample_filedir = "2b_v7/"

    year_files = {
                    "16": "combined_skim_data16__Nominal.parquet",
                    "17": "combined_skim_data17__Nominal.parquet",
                    "18": "combined_skim_data18__Nominal.parquet",
                    "22": "combined_skim_data22__Nominal.parquet",
                    "23": "combined_skim_data23__Nominal.parquet"
                    }

    for year in year_files.keys():
        Run_Over_Events(sample_filedir + year_files[year])
        # Validate_Saved_Files is for bugfixing
        # Validate_Saved_Files(sample_filedir + year_files[year], "2b2j")

    
