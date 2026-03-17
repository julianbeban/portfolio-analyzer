# test_api2.py
import requests
import json

def test_health():
    print("Testing health endpoint...")
    response = requests.get('http://localhost:5001/api/health')
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_analyze():
    print("Testing analyze endpoint...")
    response = requests.post('http://localhost:5001/api/analyze',
                            json={'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA']})
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_error_cases():
    print("Testing error cases...")
    
    # No tickers
    response = requests.post('http://localhost:5001/api/analyze', json={'tickers': []})
    print(f"Empty tickers - Status: {response.status_code}, Error: {response.json().get('error')}")
    
    # Only one ticker
    response = requests.post('http://localhost:5001/api/analyze', json={'tickers': ['AAPL']})
    print(f"Single ticker - Status: {response.status_code}, Error: {response.json().get('error')}")
    
    # Invalid ticker
    response = requests.post('http://localhost:5001/api/analyze', json={'tickers': ['AAPL', 'INVALID123']})
    print(f"Invalid ticker - Status: {response.status_code}")
    print()

if __name__ == "__main__":
    test_health()
    test_analyze()
    test_error_cases()