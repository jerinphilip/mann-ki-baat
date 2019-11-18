
import requests


response = requests.get('https://www.narendramodi.in/ka/mann-ki-baat', headers=headers, cookies=cookies)
print(response.content)

