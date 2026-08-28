import re
from pathlib import Path


def read_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content


def detect_employee_id(text):
    pattern = r"SH-\d+"
    matches = re.findall(pattern, text)

    return matches


def detect_email(text):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    matches = re.findall(pattern, text)

    return matches


def detect_percentage(text):
    pattern = r"\d+(?:\.\d+)?%"
    matches = re.findall(pattern, text)

    return matches


def detect_amount(text):
    pattern = r"₹[\d,]+"
    matches = re.findall(pattern, text)

    return matches


def sanitize_employee_id(text):
    pattern = r"SH-\d+"
    sanitized_text = re.sub(pattern, "[EMPLOYEE_ID]", text)

    return sanitized_text


def sanitize_email(text):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    sanitized_text = re.sub(pattern, "[EMAIL]", text)

    return sanitized_text


def sanitize_percentage(text):
    pattern = r"\d+(?:\.\d+)?%"
    sanitized_text = re.sub(pattern, "[PERCENTAGE]", text)

    return sanitized_text


def sanitize_amount(text):
    pattern = r"₹[\d,]+"
    sanitized_text = re.sub(pattern, "[AMOUNT]", text)

    return sanitized_text


def verify_sanitized(text):
    employee_ids = detect_employee_id(text)
    emails = detect_email(text)
    percentages = detect_percentage(text)
    amounts = detect_amount(text)

    if not employee_ids and not emails and not percentages and not amounts:
        return True

    return False


documents_folder = Path("day5_preprocessing/documents")
sanitized_folder = Path("day5_preprocessing/sanitized_documents")

sanitized_folder.mkdir(exist_ok=True)


for document_path in documents_folder.glob("*.md"):

    document_content = read_document(document_path)

    employee_ids = detect_employee_id(document_content)
    emails = detect_email(document_content)
    percentages = detect_percentage(document_content)
    amounts = detect_amount(document_content)

    sanitized_content = sanitize_employee_id(document_content)
    sanitized_content = sanitize_email(sanitized_content)
    sanitized_content = sanitize_percentage(sanitized_content)
    sanitized_content = sanitize_amount(sanitized_content)

    output_path = sanitized_folder / document_path.name

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(sanitized_content)

    if sanitized_content != document_content:
        print("\nSanitized:", document_path.name)

    verification_result = verify_sanitized(sanitized_content)

    if verification_result:
        print("Verification: PASSED", document_path.name)
    else:
        print("Verification: FAILED", document_path.name)