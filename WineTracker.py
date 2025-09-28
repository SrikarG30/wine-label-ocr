# data extraction functions
from data_extraction_functions.QRCodeScanner import scanBarcode
from data_extraction_functions.final_run_ocr import final_run_ocr
from data_extraction_functions.hybrid import final_run_blobs

# similarity functions
from similarity_functions.QRCodeSimilarity import isBarcodeSimilar
from similarity_functions.VintageSimilarity import isVintageSimilar
from similarity_functions.BlobSimilarity import isBlobDataSimilar
from similarity_functions.MakerAndCustomIDSimilarity import isMakerNameSimilar
from similarity_functions.MakerAndCustomIDSimilarity import isCustomIDSimilar
# pre-processing functions
from Photo_Stitch import stitchedImagePath


# imports
import json
import numpy as np


def run():
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
    normal_path, edge_path = stitchedImagePath(save_first_debug=True,
                                               first_debug_out="debug/first.jpg",               # base filename
                                               save_first_debug_annotated=True,
                                               first_debug_annotated_out="debug/first_annot.jpg")

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

    return record

    # Test Record Output
    # print(record)


record1 = run()
record2 = run()


#   Run test (tune thresholds as needed)  #
similar, info = isBlobDataSimilar(
    record1, record2,
    threshold=0.75,        # final decision cutoff on the 0..1 score
    pair_threshold=0.22,   # per-pair gating cost (lower = stricter)
    return_details=True
)

print("Blob similar?", similar)
print('Vintage Similar:', isVintageSimilar(record1, record2))
print('Maker Similar:', isMakerNameSimilar(record1, record2))
print('CustomID', isMakerNameSimilar(record1, record2))
print(record1)
print(record2)
