import requests
import json
from typing import Optional

# ================= CONFIG =================
API_URL = "https://searchfloxai.vercel.app/api/search"
# ==========================================

def search_floxai(query: str) -> Optional[str]:
    """
    Search real-time info using SearchFloxAI API.
    Returns formatted response or None on error.
    """
    if not query or not query.strip():
        return "Error: Query cannot be empty! Example: 'What is ArcDevs?'"

    query = query.strip()
    
    try:
        response = requests.post(
            API_URL,
            json={"query": query},
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract AI-generated answer
        answer = data.get("text", "").strip()
        
        if not answer:
            return "No result found. Try a different query!"

        # Optional: Clean up and format
        formatted = f"{answer}"
        return formatted

    except requests.exceptions.Timeout:
        return "Error: Search timed out. Try again later."
    
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to search service. Check internet."
    
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 429:
            return "Error: Too many requests. Wait a bit!"
        elif status >= 500:
            return "Error: Search service is down. Try later."
        else:
            return f"Error: HTTP {status} – Something went wrong."
    
    except json.JSONDecodeError:
        return "Error: Invalid response from server."
    
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"


# ================= USAGE EXAMPLE =================
if __name__ == "__main__":
    query = "what is ArcDevs?"
    result = search_floxai(query)
    print(result)