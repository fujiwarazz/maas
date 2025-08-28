# # StarterAPI接口：
# import requests

# heads = {
#     "accept": "application/json",
#     "X-ApiKey": "20f4de6b439cbffdb64c046d0252e2333172255b"
# }

# url = "https://wos-api.clarivate.com/api/wos/?databaseId=WOS&usrQuery=ts%3D%28duyi%29&count=10&firstRecord=1&sortField=LD%2BD&optionView=FR"
# response = requests.get(url, headers=heads)
# print(response.text)



# # ExpandedAPI接口：
# import requests
# import json
# import io
# import sys

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="gb18030")
# heads = {
#     "accept": "application/json",
#     "X-ApiKey": "20f4de6b439cbffdb64c046d0252e2333172255b"
# }

# url_new = "https://api.clarivate.com/api/wos/?databaseId=WOS&usrQuery=(UT=(WOS:001321517500001 ))&count=1&firstRecord=1"
# response = requests.get(url_new, headers=heads)
# text = response.text
# js = json.loads(text)
# print(text)


import os
import time
import clarivate.wos_starter.client
from clarivate.wos_starter.client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://api.clarivate.com/apis/wos-starter/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = clarivate.wos_starter.client.Configuration(
    host = "https://api.clarivate.com/apis/wos-starter/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ClarivateApiKeyAuth
configuration.api_key['ClarivateApiKeyAuth'] = "3349abbc61f8332387af654285930d6ac2d875ca"

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ClarivateApiKeyAuth'] = 'Bearer'


# Enter a context with an instance of the API client
with clarivate.wos_starter.client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = clarivate.wos_starter.client.DocumentsApi(api_client)
    q = 'PY=2020' # str | Web of Science advanced [advanced search query builder](https://webofscience.help.clarivate.com/en-us/Content/advanced-search.html). The supported field tags are listed in description.
    db = 'WOS' # str | Web of Science Database abbreviation * WOS - Web of Science Core collection * BIOABS - Biological Abstracts * BCI - BIOSIS Citation Index * BIOSIS - BIOSIS Previews * CCC - Current Contents Connect * DIIDW - Derwent Innovations Index * DRCI - Data Citation Index * MEDLINE - MEDLINE The U.S. National Library of Medicine® (NLM®) premier life sciences database. * ZOOREC - Zoological Records * PPRN - Preprint Citation Index * WOK - All databases  (optional) (default to 'WOS')
    limit = 10 # int | set the limit of records on the page (1-50) (optional) (default to 10)
    page = 1 # int | set the result page (optional) (default to 1)
    sort_field = 'LD+D' # str | Order by field(s). Field name and order by clause separated by '+', use A for ASC and D for DESC, ex: PY+D. Multiple values are separated by comma. Supported fields:  * **LD** - Load Date * **PY** - Publication Year * **RS** - Relevance * **TC** - Times Cited  (optional)
    modified_time_span = None # str | Defines a date range in which the results were most recently modified. Beginning and end dates must be specified in the yyyy-mm-dd format separated by '+' or ' ', e.g. 2023-01-01+2023-12-31. This parameter is not compatible with the all databases search, i.e. db=WOK is not compatible with this parameter. (optional)
    tc_modified_time_span = None # str | Defines a date range in which times cited counts were modified. Beginning and end dates must be specified in the yyyy-mm-dd format separated by '+' or ' ', e.g. 2023-01-01+2023-12-31. This parameter is not compatible with the all databases search, i.e. db=WOK is not compatible with this parameter. (optional)
    detail = None # str | it will returns the full data by default, if detail=short it returns the limited data (optional)

    try:
        # Query Web of Science documents 
        api_response = api_instance.documents_get(q, db=db, limit=limit, page=page, sort_field=sort_field, modified_time_span=modified_time_span, tc_modified_time_span=tc_modified_time_span, detail=detail)
        print("The response of DocumentsApi->documents_get:\n")
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling DocumentsApi->documents_get: %s\n" % e)
