# data extraction functions
from data_extraction_functions.QRCodeScanner import scanBarcode
from data_extraction_functions.final_run_ocr import final_run_ocr
# from data_extraction_functions.hybrid import final_run_blobs -> Comment Once Added to Repo

# similarity functions
from similarity_functions.QRCodeSimilarity import isBarcodeSimilar
from similarity_functions.VintageSimilarity import isVintageSimilar

# pre-processing functions
from Photo_Stitch import stitchedImagePath


# imports
import json
import numpy as np


# Initialize record with empty data
record = {
    'CustomID': None,       # string ('MakerName|Vintage')
    'MakerName': None,      # string
    'Vintage': None,        # integer (4 digits) (can also be string)
    'Barcode': None,        # string
    'BlobData': {}          # json (coordinates)
}


# -------------------- RUN QR -------------------- #
barcode = scanBarcode(0)

# -------- Get File Path for OCR and Blob -------- #
normalFramePath, edgeFramePath = stitchedImagePath()


# ------------------- RUN OCR ------------------- #
if normalFramePath:
    custom_id, maker, vintage = final_run_ocr(normalFramePath, "weights.pt")

# ------------------- RUN BLOB ------------------ #
# ---- Uncomment Once Blob Function Imported ---- #
# if edgeFramePath:
#     blobData = final_run_blobs(edgeFramePath)

# ------------ Assign Record Entries ------------ #
if barcode:
    record['Barcode'] = str(barcode)
if custom_id is not None:
    record["CustomID"] = custom_id
if maker:
    record["MakerName"] = maker
if vintage is not None:
    record["Vintage"] = vintage
# --- Uncomment Once Blob Function Imported ---- #
# if blobData is not None:
#     record['BlobData'] = blobData



# Test Record Output
print(record)
