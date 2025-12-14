from pathlib import Path
from typing import Dict, Any, List, Optional
import json


def load_data(data_dir: Path = Path('.')) -> Dict[str, Any]:
    """Load insurance policies and reference code mappings from JSON files."""

    insurance_policies_file = data_dir / "policies.json"
    reference_codes_file = data_dir / "diagnosis_codes.json"

    with open(insurance_policies_file, "r") as fh:
        insurance_policies = json.load(fh)

    with open(reference_codes_file, "r") as fh:
        reference_codes = json.load(fh)

    return {
        "insurance_policies": insurance_policies,
        "reference_codes": reference_codes,
    }


def get_policy_by_id(policy_id: str, policies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the policy dict for `policy_id` or None if not found."""
    
    for policy in policies:
        if policy.get("policy_id") == policy_id:
            return policy
    return None
