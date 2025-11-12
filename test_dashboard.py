#!/usr/bin/env python3
"""
Test script for the dashboard API endpoints
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        print(f"Testing {method} {endpoint}...")
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"✅ Success: {response.status_code}")
            return response.json()
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print("Make sure the FastAPI server is running on port 8000")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🧪 Testing Iris Dashboard API Endpoints")
    print("=" * 50)
    
    # Test basic endpoints
    endpoints_to_test = [
        "/dashboard/health",
        "/dashboard/summary", 
        "/dashboard/users",
        "/dashboard/analytics?days=7",
        "/dashboard/queries?limit=5",
        "/dashboard/sessions"
    ]
    
    for endpoint in endpoints_to_test:
        result = test_endpoint(endpoint)
        if result:
            print(f"   Response keys: {list(result.keys()) if isinstance(result, dict) else 'List with ' + str(len(result)) + ' items'}")
        print()
    
    # Test creating a test user
    print("Testing user creation...")
    test_user_data = {
        "user_id": "test_dashboard_user",
        "username": "Dashboard Test User",
        "email": "test@dashboard.com",
        "daily_query_limit": 50,
        "is_admin": False
    }
    
    result = test_endpoint("/dashboard/users", "POST", test_user_data)
    if result:
        print(f"   Created user: {result.get('user_id')}")
    print()
    
    # Test rate limit check
    print("Testing rate limit check...")
    result = test_endpoint("/dashboard/rate-limit/test_dashboard_user")
    if result:
        print(f"   Rate limit status: {result}")
    print()
    
    print("=" * 50)
    print("✅ Dashboard API testing completed!")
    print("\nTo start the dashboard interface:")
    print("streamlit run app/dashboard_app.py --server.port 8502")

if __name__ == "__main__":
    main()
