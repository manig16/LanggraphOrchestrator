import json
from datetime import date
from typing import Dict, Any


def calculate_age(dateOfBirth: str, dateOfService: str) -> int | None:
    try:
        dob = date.fromisoformat(dateOfBirth)
        dos = date.fromisoformat(dateOfService)
    except (ValueError, TypeError):
        return None

    age = dos.year - dob.year
    if dos.month < dob.month or (dos.month == dob.month and dos.day < dob.day):
        age -= 1
    return age


def run_validation_loop(data_path: str, process_claim_fn, app) -> None:
    with open(data_path, 'r') as f:
        records = json.load(f)

    print(f"Num records to process: {len(records)}")
    for record in records:
        patient_id = record.get("patient_id")
        age = calculate_age(record.get("date_of_birth"), record.get("date_of_service"))
        record["age"] = age

        try:
            generated_response = process_claim_fn(record, app)
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            generated_response = f"Error processing patient {patient_id}: {e}"

        print(f"{patient_id}, {age}, {generated_response}")
