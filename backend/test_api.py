import requests
import json

# Test the analyze endpoint
response = requests.post('http://localhost:5001/api/analyze', 
                        json={'tickers': ['AAPL', 'MSFT', 'GOOGL']})

# Pretty print the JSON response
print(json.dumps(response.json(), indent=2))