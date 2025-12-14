import json
from typing import Dict, Any

from langchain_core.tools import tool

LLM_CLIENT = None
REFERENCE_CODES_DATA = None

def set_llm_client(client: object) -> None:
    global LLM_CLIENT
    LLM_CLIENT = client


def set_reference_codes(reference_data: Dict[str, Any]) -> None:
    global REFERENCE_CODES_DATA
    REFERENCE_CODES_DATA = reference_data


@tool
def summarize_patient_record(record_str) -> str:
    """Generate a structured summary of a patient claim record using the LLM."""
    global LLM_CLIENT, REFERENCE_CODES_DATA
    prompt = f"""You are a health insurance claim summarization agent. Read the patient record and the reference codes. Then
    generate a summary clearly formatted and include the following seven labeled sections, in the specified order:
        - Patient Demographics: Include: name, gender, age 
        - Insurance Policy ID
        - Diagnoses and Descriptions: Include ICD10 codes and their mapped descriptions. 
        - Procedures and Descriptions: Include CPT codes and their mapped descriptions. 
        - Preauthorization Status: Clearly mention if preauthorization was required and whether it was obtained.
        - Billed Amount (in USD) 
        - Date of Service

    Use the provided ICD-10 and CPT code mappings (from REFERENCE_CODES_DATA) to enrich the output 
    with human-readable descriptions of medical codes.

    The summary text should follow a bullet-point format or clearly separated labeled sections.

    The patient record is: {record_str}
    The reference codes are: {REFERENCE_CODES_DATA}
    """
    return LLM_CLIENT.invoke(prompt).content


@tool
def summarize_policy_guideline(policy_id: str) -> str:
    """Summarize the insurance policy for a given policy id using the LLM."""
    global LLM_CLIENT, REFERENCE_CODES_DATA
    from loader import get_policy_by_id, load_data 

    policy_data = None
    if REFERENCE_CODES_DATA and "insurance_policies" in REFERENCE_CODES_DATA:
        policy_data = get_policy_by_id(policy_id, REFERENCE_CODES_DATA["insurance_policies"])

    if policy_data is None:
        return f"Error: {policy_id} not found"

    prompt = f"""You are a health insurance policy agent. Read the policy data and generate a summary to include the following  labeled sections, in order:
        - Policy Details: Include: policy ID and plan name 
        - Covered Procedures: For each covered procedure listed in the policy, include the following sub-points:
            * Procedure Code and Description (using CPT code mappings) 
            * Covered Diagnoses and Descriptions: List ALL diagnosis codes covered for this procedure (using ICD-10 code mappings)
            * Gender Restriction
            * Age Range (specify the exact lower and upper bounds)
            * Preauthorization Requirement
            * Notes on Coverage (if any) 

        The policy data is: {policy_data}
        The reference codes are: {REFERENCE_CODES_DATA}
    """
    return LLM_CLIENT.invoke(prompt).content


@tool
def check_claim_coverage(record_summary: str, policy_summary: str) -> str:
    """Determine whether procedures in the record are covered by the policy."""
    global LLM_CLIENT
    if (record_summary is None or policy_summary is None or record_summary == "" or policy_summary == ""):
        return "Error in check_claim_coverage"

    prompt = f"""You are a claim validation agent. Perform a step by step evaluation of the patient record with 
    the policy summary.

    Coverage Evaluation Criteria: 
    A procedure should be approved only if ALL the following five conditions are met:
        1. Diagnosis Match: At least ONE of the patient's diagnosis codes MUST match ANY of the policy-covered diagnoses for the claimed procedure.
        2. The procedure code must explicitly listed in the policy, and all associated conditions are satisfied.
        3. The patient's age must fall within the policy's defined age range (inclusive of the lower bound, exclusive of the upper bound).
        4. The patient's gender matches the policy’s requirement for that procedure.
        5. If preauthorization is required by the policy, it must have been obtained.

    Expected Output Format: The response should include the following three sections: 
        1. Coverage Review
        2. Summary of Findings
        3. Final Decision: For each procedure for the claim, return either "APPROVE" or "REVIEW REQUIRED" with a brief explanation.

    Patient record summary: {record_summary}
    Policy summary: {policy_summary}
    """
    return LLM_CLIENT.invoke(prompt).content
