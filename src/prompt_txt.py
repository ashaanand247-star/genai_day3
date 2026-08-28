PROMPT_V1 = """
Summarize the following news in one paragraph.

News:
{text}
"""

PROMPT_V2 = """
Summarize the following news in exactly three bullet points.
Mention important people, places, and dates.

News:
{text}
"""

test_cases = [
    {
        "name": "Normal News",
        "input": "India launched a new satellite successfully today."
    },
    {
        "name": "Empty Input",
        "input": ""
    },
    {
        "name": "Long News",
        "input": "Paste a long news article here."
    }
]

for case in test_cases:
    print("=" * 50)
    print("Test:", case["name"])

    print("\nPrompt V1")
    print(PROMPT_V1.format(text=case["input"]))

    print("\nPrompt V2")
    print(PROMPT_V2.format(text=case["input"]))