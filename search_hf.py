import requests
response = requests.get('https://huggingface.co/api/datasets?search=nuclei')
for ds in response.json()[:10]:
    print(ds['id'])
