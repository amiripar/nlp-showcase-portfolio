from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import requests

def unwrap_api_response(resp: requests.Response) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        body = resp.json()
    except Exception:
        return False, None, None

    if not isinstance(body, dict):
        return False, None, None

    ok = bool(body.get("ok", False))
    if ok:
        return True, body.get("data"), body.get("meta")
    return False, None, body.get("meta")

def extract_error(resp: requests.Response) -> Dict[str, Any]:
    try:
        body = resp.json()
        if isinstance(body, dict) and "error" in body:
            return body["error"]
    except Exception:
        pass

    return {
        "code": f"http_{resp.status_code}",
        "message": (resp.text[:500] if resp.text else "Request failed"),
        "details": None,
    }
