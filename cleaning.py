from pathlib import Path


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()


    return content


def clean_text(text):
    lines = text.splitlines()

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        line = line.strip()

        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines)


sanitized_folder = Path("day5_preprocessing/sanitized_documents")
cleaned_folder = Path("day5_preprocessing/cleaned_documents")
cleaned_folder.mkdir(exist_ok=True)


for document_path in sanitized_folder.glob("*.md"):
    document_content = load_document(document_path)
    cleaned_content = clean_text(document_content)

    output_path = cleaned_folder / document_path.name

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_content)

    print(document_path.name)