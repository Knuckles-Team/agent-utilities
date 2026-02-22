
import os
import sys
import requests
from dotenv import load_dotenv

# Add parent directory to path to import adguard_home_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adguard_home_agent.adguard_api import Api
import adguard_home_agent.adguard_api
print(f"DEBUG: adguard_api imported from {adguard_home_agent.adguard_api.__file__}")

load_dotenv()

BASE_URL = os.getenv("ADGUARD_URL")
USERNAME = os.getenv("ADGUARD_USERNAME")
PASSWORD = os.getenv("ADGUARD_PASSWORD")

print(f"Testing connection to: {BASE_URL}")
print(f"Username: {USERNAME}")

def test_custom_api_class():
    print("\n--- Testing adguard_api.Api class ---")
    # New Api init signature: base_url, username, password, verify, proxies
    api = Api(base_url=BASE_URL, username=USERNAME, password=PASSWORD)
    try:
        # get_version now calls /control/status
        version = api.get_version()
        print(f"Success! Version/Status: {version}")
    except Exception as e:
        print(f"Failed to get version via Api class: {e}")

    try:
        print("Attempting get_stats()...")
        stats = api.get_stats()
        print("Success! Stats retrieved.")
        # print keys to verify
        print(f"Stats keys: {list(stats.keys())}")
    except Exception as e:
        print(f"Failed to get stats via Api class: {e}")

    try:
        print("Attempting list_clients()...")
        clients = api.list_clients()
        print(f"Clients content: {clients}")
        print(f"Clients response type: {type(clients)}")
        if isinstance(clients, dict):
             client_list = clients.get('clients') or []
             print(f"Success! Clients retrieved. Count: {len(client_list)}")
        elif isinstance(clients, list):
             print(f"Success! Clients retrieved. Count: {len(clients)}")
        else:
             print(f"Clients retrieved but unknown format: {clients}")
    except Exception as e:
        print(f"Failed to list clients via Api class: {e}")

    try:
        print("Attempting get_filtering_status()...")
        filtering = api.get_filtering_status()
        print(f"Success! Filtering status retrieved via Api class.")
        print(f"Filtering enabled: {filtering.get('enabled')}")
    except Exception as e:
        print(f"Failed to get filtering status via Api class: {e}")

    return api

def test_standard_endpoints(client):
    print("\n--- Testing Expanded APIs (Read-Only) ---")
    try:
        print("Attempting get_dns_info()...")
        dns_info = client.get_dns_info()
        print(f"Success! DNS Info keys: {list(dns_info.keys())}")
    except Exception as e:
        print(f"Error in get_dns_info: {e}")

    try:
        print("Attempting get_dhcp_interfaces()...")
        dhcp_interfaces = client.get_dhcp_interfaces()
        print(f"Success! DHCP Interfaces found: {len(dhcp_interfaces)}")
    except Exception as e:
        print(f"Error in get_dhcp_interfaces: {e}")

    try:
        print("Attempting get_all_blocked_services()...")
        blocked_services = client.get_all_blocked_services()
        print(f"Success! Blocked services retrieved. Count: {len(blocked_services.get('blocked_services', []))}")
    except Exception as e:
        print(f"Error in get_all_blocked_services: {e}")

    try:
        print("Attempting get_profile()...")
        profile = client.get_profile()
        print(f"Success! Profile retrieved for: {profile.get('name', 'unknown')}")
    except Exception as e:
        print(f"Error in get_profile: {e}")

    try:
        print("Attempting get_rewrite_settings()...")
        rewrites = client.get_rewrite_settings()
        print(f"Success! Rewrite settings retrieved.")
    except Exception as e:
        print(f"Error in get_rewrite_settings: {e}")

    try:
        print("Attempting get_tls_status()...")
        tls_status = client.get_tls_status()
        print(f"Success! TLS status retrieved.")
    except Exception as e:
        print(f"Error in get_tls_status: {e}")

    try:
        print("Attempting search_clients()...")
        # Search for something unlikely to exist or just empty query if allowed?
        # Based on my code, query is required. I'll search for "127.0.0.1" which returned [] earlier.
        search_result = client.search_clients(query="127.0.0.1")
        print(f"Success! Search result type: {type(search_result)}")
    except Exception as e:
        print(f"Error in search_clients: {e}")

    print("\n--- Testing Standard AdGuard Home Endpoints (/control/...) ---")
    # AdGuard Home usually uses /control/api or similar.
    # Common endpoints: /control/status, /control/version.json (sometimes)

    session = requests.Session()
    # AdGuard Home often uses Basic Auth
    session.auth = (USERNAME, PASSWORD)

    endpoints = [
        "/control/status",
        "/control/version.json",
        "/control/stats",
        "/control/dns_config",
        "/control/filtering/status",
        "/control/access/list"
    ]

    for endpoint in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            print(f"Requesting {url} ...")
            resp = session.get(url)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"Response (truncated): {resp.text[:100]}")
            else:
                 print(f"Response: {resp.text[:100]}")
        except Exception as e:
            print(f"Error requesting {url}: {e}")

def test_login_cookie():
    print("\n--- Testing Cookie-based Login (/control/login) ---")
    url = f"{BASE_URL}/control/login"
    try:
        resp = requests.post(url, json={"name": USERNAME, "password": PASSWORD})
        print(f"Login Status: {resp.status_code}")
        if resp.status_code == 200:
            print("Login successful! received cookies.")
            print(resp.cookies)
        else:
            print(f"Login failed: {resp.text}")
    except Exception as e:
        print(f"Error during login: {e}")

if __name__ == "__main__":
    client = test_custom_api_class()
    if client:
        test_standard_endpoints(client)
    test_login_cookie()
