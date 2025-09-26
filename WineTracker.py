# functions
from QRCodeScanner import scanBarcode
from extract_maker_vintage import wine_fields

# imports
import json




# Initialize record with empty data
record = {
    'CustomID': None,       # string ('MakerName|Vintage')
    'MakerName': None,      # string
    'Vintage': None,        # integer (4 digits)
    'Barcode': None,        # 
    'BlobData': None        # json (coordinates)
}


# Run QRCodeScanner
barcode = scanBarcode(0)
if barcode != 0:
    record['Barcode'] = barcode

custom_id, maker_name, vintage = wine_fields('workflowTest.jpeg', 'weights.pt')

if custom_id:
    record['CustomID'] = custom_id
    record['MakerName'] = maker_name
    record['Vintage'] = vintage
elif maker_name:
    record['MakerName'] = maker_name
elif vintage:
    record['Vintage'] = vintage

print(record)

