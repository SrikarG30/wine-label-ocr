# data extraction functions
from data_extraction_functions.QRCodeScanner import scanBarcode
from data_extraction_functions.final_run_ocr import final_run_ocr

# similarity functions
from similarity_functions.QRCodeSimilarity import isBarcodeSimilar
from similarity_functions.VintageSimilarity import isVintageSimilar


# imports
import json



# Initialize record with empty data
record = {
    'CustomID': None,       # string ('MakerName|Vintage')
    'MakerName': None,      # string
    'Vintage': None,        # integer (4 digits) (can also be string)
    'Barcode': None,        # string
    'BlobData': {}          # json (coordinates)
}



# --------- RUN QR CODE SCANNER ---------
# comment out these three lines if pyzbar not working to test other code
# barcode = scanBarcode(0)
# if barcode:
#     record['Barcode'] = str(barcode)

# uncomment line below if pyzbar not working to test other code
record['Barcode'] = 'temp' 



# --------- RUN QR CODE SCANNER ---------
img = "test_images/009.jpg" # right now using a static image for testing purposes
# img = get_wine_label() -> Function will return stitched image of wine label

# Armando will integrate image stitching
#   -> Live Camera will open
#   -> Stitched image of wine label is returned
#   -> Image used for both OCR and Blob



# --------- RUN PADDLEOCR/ROBOFLOW ---------
custom_id, maker, vintage = final_run_ocr("test_images/009.jpg", "weights.pt")

# Set fields if present
if custom_id is not None:
    record["CustomID"] = custom_id
if maker:
    record["MakerName"] = maker
if vintage is not None:
    record["Vintage"] = vintage



# Run Blob Method Here 
# extract_blob_data should return a json or a dictionary
# record['BlobData'] = extract_blob_data('image path')



# Output record for testing
print(record)

