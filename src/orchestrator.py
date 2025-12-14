import json
import traceback
from typing import TypedDict, Dict, Any, Optional

from tools import summarize_patient_record, summarize_policy_guideline, check_claim_coverage
from langgraph.graph import StateGraph, START, END


class ClaimApprovalState(TypedDict):
    record: Dict[str, Any]
    record_summary: str
    policy_summary: str
    coverage_report: str
    final_response: str
    tool_step: int


tool_map = {
    "summarize_patient_record": summarize_patient_record,
    "summarize_policy_guideline": summarize_policy_guideline,
    "check_claim_coverage": check_claim_coverage,
}


def agent_orchestrator(state: ClaimApprovalState) -> Dict[str, Any]:
    current_step = state["tool_step"]
    record = state["record"]
    updates: Dict[str, Any] = {"tool_step": current_step + 1}

    record_str = json.dumps(record)

    try:
        if current_step == 0:
            summary = summarize_patient_record.invoke({"record_str": record_str})
            updates["record_summary"] = summary
        elif current_step == 1:
            policy_id = record.get("insurance_policy_id")
            summary = summarize_policy_guideline.invoke({"policy_id": policy_id})
            updates["policy_summary"] = summary
        elif current_step == 2:
            record_summary = state.get("record_summary")
            policy_summary = state.get("policy_summary")
            updates["coverage_report"] = check_claim_coverage.invoke({
                "record_summary": record_summary,
                "policy_summary": policy_summary,
            })
        else:
            raise ValueError(f"Invalid tool step: {current_step}")
    except Exception as e:
        if current_step == 0:
            updates["record_summary"] = "Error: " + str(e)
        elif current_step == 1:
            updates["policy_summary"] = "Error: " + str(e)
        elif current_step == 2:
            updates["coverage_report"] = "Error: " + str(e)

        print(f"Error in tool step {current_step}: {e}")
        traceback.print_exc()

    return updates


def agent_reasoning(state: ClaimApprovalState) -> Dict[str, str]:
    from auth import LLM_CLIENT

    final_prompt = f""" {state} \nNow, extract the final decision and the concise reason for the decision, and present it in the following format:\nDecision: [APPROVE or REVIEW REQUIRED]\nReason: [A concise reason based on the tool's findings]."""

    try:
        final_output = LLM_CLIENT.invoke(final_prompt).content
        return {"final_response": final_output}
    except Exception as e:
        print(f"Error in final reasoning: {e}")
        traceback.print_exc()
        return {"final_response": "Error: " + str(e)}


def build_workflow() -> Any:
    workflow = StateGraph(ClaimApprovalState)
    workflow.add_node("summarize_patient_record_node", agent_orchestrator)
    workflow.add_node("summarize_policy_guideline_node", agent_orchestrator)
    workflow.add_node("check_claim_coverage_node", agent_orchestrator)
    workflow.add_node("final_reasoning", agent_reasoning)

    workflow.add_edge(START, "summarize_patient_record_node")
    workflow.add_edge("summarize_patient_record_node", "summarize_policy_guideline_node")
    workflow.add_edge("summarize_policy_guideline_node", "check_claim_coverage_node")
    workflow.add_edge("check_claim_coverage_node", "final_reasoning")
    workflow.add_edge("final_reasoning", END)

    return workflow.compile()


def process_claim(record: Dict[str, Any], app: Any) -> str:
    initial_state: ClaimApprovalState = {
        "record": record,
        "record_summary": "",
        "policy_summary": "",
        "coverage_report": "",
        "final_response": "",
        "tool_step": 0,
    }

    final_state = app.invoke(initial_state)

    if "final_response" in final_state and final_state["final_response"]:
        return final_state["final_response"]
    else:
        return final_state.get("coverage_report", "Error: No response generated")
