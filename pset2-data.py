from bs4 import BeautifulSoup
import urllib.request
import urllib.error

# Gather data and save (hard-coded). Try out beautiful soup!

url = 'http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html'
output_filename = 'heights.txt'

try:
    with urllib.request.urlopen(url) as response:
        s = response.read()
except urllib.error.URLError as e:
    print(e.reason)
    sys.exit(1)

soup = BeautifulSoup(s)

with open(output_filename, 'w') as f:
    for row in soup.find_all('tr'):
        # hard coded html formats wooooo
        height_value = row.contents[3].string
        f.write(height_value + '\n')
