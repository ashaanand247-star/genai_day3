import json

from model_client import ModelClient
from validator import validate_response
from models import SummarizationOutput


def load_test_cases():
    with open("src/datasets/prompt_test_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results):
    with open("src/outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main():

    test_cases = load_test_cases()

    prompt_version = "v1"
    prompt_file = "src/prompts/summarization.txt"

    client = ModelClient()

    results = []

    for case in test_cases:

        print(f"Running {case['id']}")

        # Read input file
        try:
            with open("src/" + case["input_file"], "r", encoding="utf-8") as f:
                article = f.read()

        except FileNotFoundError as e:

            results.append({
                "case_id": case["id"],
                "task": case["task"],
                "prompt_version": prompt_version,
                "model_version": "N/A",
                "latency": 0,
                "validation_result": "Not Executed",
                "failure_category": "File Error",
                "failure_reason": str(e),
                "status": "FAIL",
                "generated_text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })

            print(f"File Error in {case['id']}: {e}")
            continue

        # Read prompt template
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        # Replace placeholder with article
        system_prompt = system_prompt.replace("{{TEXT}}", article)

        # Generate AI response
        try:
            result = client.generate(system_prompt, "")

        except Exception as e:

            results.append({
                "case_id": case["id"],
                "task": case["task"],
                "prompt_version": prompt_version,
                "model_version": "N/A",
                "latency": 0,
                "validation_result": "Not Executed",
                "failure_category": "Network/Model Error",
                "failure_reason": str(e),
                "status": "FAIL",
                "generated_text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })

            print(f"Network/Model Error in {case['id']}: {e}")
            continue

        # Use the actual model response for validation
        response_data = {
            "summary": result["text"]
        }

        # Validate response
        validated_data, error = validate_response(
            response_data,
            SummarizationOutput
        )
        # Determine failure category
        if error:
            failure_category = "Validation Error"
            failure_reason = error
            status = "FAIL"
        else:
            failure_category = "None"
            failure_reason = "None"
            status = "PASS"

        # Save result
        results.append({
            "case_id": case["id"],
            "task": case["task"],
            "prompt_version": prompt_version,
            "model_version": result["model"],
            "latency": result["latency"],
            "validation_result": "Success" if error is None else "Failed",
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "status": status,
            "generated_text": result["text"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"]
        })

    save_results(results)

    print("\nAll test cases completed!")
    print("Results saved to src/outputs/results.json")


if __name__ == "__main__":
    main()