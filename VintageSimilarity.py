# 1:1 check of Vintage from 2 records

# Record Format for reference
# record = {
#     'CustomID': None,       # string ('MakerName|Vintage')
#     'MakerName': None,      # string
#     'Vintage': None,        # integer (4 digits) (can also be string)
#     'Barcode': None,        # string
#     'BlobData': {}          # json (coordinates)
# }

def isVintageSimilar(r1: dict, r2: dict) -> bool:
    v1 = r1.get('Vintage')
    v2 = r2.get('Vintage')
    if not v1 or not v2: # If either record doesn't have a Vintage
        return False
    return v1 == v2 # check if Vtinages are identical