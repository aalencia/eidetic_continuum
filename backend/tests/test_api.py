import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_humanitarian_exception():
    print("\n--- Running Humanitarian Exception Test ---")
    transaction_data = {
        "transaction_amount": 120000,
        "destination_country_risk": "sanctioned",
        "purpose": "medical_aid",
        "security_risk_level": "low",
    }
    
    try:
        response = requests.post(f"{BASE_URL}/evaluate", json=transaction_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()

        print("Request Data:", json.dumps(transaction_data, indent=2))
        print("Response Data:", json.dumps(result, indent=2))

        assert result["decision"] == "APPROVE with enhanced documentation"
        assert "override" in result["reasoning"]
        print("Test Passed: Humanitarian Exception handled correctly.")

    except requests.exceptions.RequestException as e:
        print(f"Test Failed: Could not connect to the backend. Error: {e}")
    except AssertionError:
        print("Test Failed: Unexpected decision or reasoning.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def test_clear_violation():
    print("\n--- Running Clear Violation Test ---")
    transaction_data = {
        "transaction_amount": 5000,
        "destination_country_risk": "low_risk",
        "purpose": "sanctioned_activity",
        "security_risk_level": "high",
    }

    try:
        response = requests.post(f"{BASE_URL}/evaluate", json=transaction_data)
        response.raise_for_status()
        result = response.json()

        print("Request Data:", json.dumps(transaction_data, indent=2))
        print("Response Data:", json.dumps(result, indent=2))

        assert result["decision"] == "Rejection"
        assert "Clear constitutional violation" in result["reasoning"]
        print("Test Passed: Clear Violation handled correctly.")

    except requests.exceptions.RequestException as e:
        print(f"Test Failed: Could not connect to the backend. Error: {e}")
    except AssertionError:
        print("Test Failed: Unexpected decision or reasoning.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def test_get_decision_history():
    print("\n--- Running Get Decision History Test ---")
    # First, make a few decisions to populate the history
    test_humanitarian_exception()
    time.sleep(0.1) # Give a small delay to ensure distinct timestamps
    test_clear_violation()
    time.sleep(0.1)

    try:
        response = requests.get(f"{BASE_URL}/get-decision-history")
        response.raise_for_status()
        history = response.json()

        print("Decision History:", json.dumps(history, indent=2))

        assert isinstance(history, list)
        assert len(history) >= 2 # Should have at least the two decisions we just made
        
        # Check structure of a history entry
        if history:
            first_entry = history[0]
            assert "timestamp" in first_entry
            assert "transaction_input" in first_entry
            assert "decision_output" in first_entry
            assert "decision" in first_entry["decision_output"]

        print("Test Passed: Decision history retrieved and structured correctly.")

    except requests.exceptions.RequestException as e:
        print(f"Test Failed: Could not connect to the backend. Error: {e}")
    except AssertionError:
        print("Test Failed: Unexpected decision history content or structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Clear history before running tests if possible, or ensure tests are idempotent
    # For this simple in-memory list, restarting the backend clears it.
    # We'll just run the tests sequentially.
    test_humanitarian_exception()
    test_clear_violation()
    test_get_decision_history()
