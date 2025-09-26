# 1:1 check of QR Code from 2 records

# Record Format for reference
# record = {
#     'CustomID': None,       # string ('MakerName|Vintage')
#     'MakerName': None,      # string
#     'Vintage': None,        # integer (4 digits) (can also be string)
#     'Barcode': None,        # string
#     'BlobData': {}          # json (coordinates)
# }


def isBarcodeSimilar(r1: dict, r2: dict) -> bool:
    b1 = r1.get('Barcode')
    b2 = r2.get('Barcode')
    if not b1 or not b2: # If either record doesn't have a barcode
        return False
    return b1 == b2 # check if barcodes are identical


# TESTING:
# record_a = {
#     "CustomID": "OpusOne|2018",
#     "MakerName": "Opus One",
#     "Vintage": 2018,
#     "Barcode": "123456789012",   # EAN-13 (leading 0)
#     "BlobData": {}
# }

# record_b = {
#     "CustomID": "OpusOne|2018",
#     "MakerName": "Opus One",
#     "Vintage": 2018,
#     "Barcode": "123456789012",    # UPC-A (12 digits), same code family
#     "BlobData": {}
# }

# print(isBarcodeSimilar(record_a, record_b))