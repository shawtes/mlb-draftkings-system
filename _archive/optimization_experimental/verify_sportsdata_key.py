#!/usr/bin/env python3
"""
Verify sportsdata.io API key and find correct endpoints
"""

import requests

def test_api_key(api_key):
    """Test API key with various endpoints to find what works"""
    
    print("="*70)
    print("SportsData.io API Key Verification")
    print("="*70)
    print(f"\n🔑 Testing API Key: {api_key[:20]}...{api_key[-4:]}\n")
    
    # Test with different headers formats
    header_formats = [
        {"Ocp-Apim-Subscription-Key": api_key},
        {"Authorization": f"Bearer {api_key}"},
        {"X-API-Key": api_key},
        {"apiKey": api_key},
    ]
    
    # Simple endpoint to test
    test_url = "https://api.sportsdata.io/v3/nfl/scores/json/AreAnyGamesInProgress"
    
    print("Testing different authentication methods:\n")
    
    for i, headers in enumerate(header_formats, 1):
        header_name = list(headers.keys())[0]
        print(f"{i}. Testing with header: {header_name}")
        
        try:
            response = requests.get(test_url, headers=headers, timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS! This header format works!")
                print(f"   Response: {response.text[:100]}")
                return headers
            elif response.status_code == 401:
                print(f"   ❌ 401: Invalid key or wrong header format")
            elif response.status_code == 403:
                print(f"   ⚠️  403: Key valid but no access to this endpoint")
                return headers  # Key is valid but wrong product
            else:
                print(f"   ❌ {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
        print()
    
    print("\n" + "="*70)
    print("❌ API Key Verification Failed")
    print("="*70)
    print("\n📝 Please check:")
    print("   1. Go to: https://sportsdata.io/members/subscriptions")
    print("   2. Verify your API key is correct")
    print("   3. Check which NFL product you're subscribed to:")
    print("      - NFL Scores & Schedules")
    print("      - NFL Stats")  
    print("      - NFL Projections")
    print("      - NFL DFS")
    print("   4. Make sure the subscription is active")
    print("   5. Some keys are for different sports (NBA, MLB, etc.)")
    print("\n💡 Tip: Copy the key directly from your dashboard")
    print("   Don't include any spaces or extra characters\n")
    
    return None

if __name__ == "__main__":
    # Get API key from user or file
    import os
    import sys
    
    api_key = os.environ.get('SPORTSDATA_API_KEY')
    
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        api_key = input("Enter your sportsdata.io API key: ").strip()
    
    if api_key:
        valid_headers = test_api_key(api_key)
        
        if valid_headers:
            print("\n✅ API Key is working!")
            print(f"   Use these headers: {valid_headers}")
        else:
            print("\n❌ Could not verify API key")
            print("   Please check your sportsdata.io account")
    else:
        print("No API key provided")


