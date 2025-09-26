# functions
from QRCodeScanner import scanBarcode
from final_run_ocr import final_run_ocr

# imports
import json




# Initialize record with empty data
record = {
    'CustomID': None,       # string ('MakerName|Vintage')
    'MakerName': None,      # string
    'Vintage': None,        # integer (4 digits)
    'Barcode': None,        # 
    'BlobData': {}        # json (coordinates)
}


# Run QRCodeScanner
barcode = scanBarcode(0)
if barcode:
    record['Barcode'] = str(barcode)

custom_id, maker, vintage = final_run_ocr("workflowTest.jpeg", "weights.pt")

# Set fields if present
if custom_id is not None:
    record["CustomID"] = custom_id
if maker:
    record["MakerName"] = maker
if vintage is not None:
    record["Vintage"] = vintage

print(record)

