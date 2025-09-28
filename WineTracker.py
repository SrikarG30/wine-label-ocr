# data extraction functions
from data_extraction_functions.QRCodeScanner import scanBarcode
from data_extraction_functions.final_run_ocr import final_run_ocr
from data_extraction_functions.hybrid import final_run_blobs

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
normal_path, edge_path = stitchedImagePath()


# ------------------- RUN OCR ------------------- #
if normal_path:
    custom_id, maker, vintage = final_run_ocr(normal_path, "weights.pt")

# ------------------- RUN BLOB ------------------ #
if edge_path:
    blobData = final_run_blobs(
        image_path=edge_path,
        use_image_as_mask=True,     # if your prefilter produced “mostly masked” image
        crop_label=False,           # keep full frame unless you want detector crop
        skip_alignment=True,        # turntable = already straight
        database="wine_database.jsonl",
        crop_weights="crop_weights.pt",
        debug_out="debug"           # or None
    )

# ------------ Assign Record Entries ------------ #
if barcode:
    record['Barcode'] = str(barcode)
if custom_id is not None:
    record["CustomID"] = custom_id
if maker:
    record["MakerName"] = maker
if vintage is not None:
    record["Vintage"] = vintage
if blobData is not None:
    record['BlobData'] = blobData


# Test Record Output
print(record)
