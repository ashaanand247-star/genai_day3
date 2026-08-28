# Prompt Version Comparison

## Objective

Compare two summarization prompt versions using the same fixed 10-case test dataset and the same model.

## Model

nvidia/nemotron-3-nano-30b-a3b:free

## Prompt Versions

### V1

```text
You are an expert summarizer.

Task:
Summarize the following text in one short paragraph.

Rule:
Do not add any information that is not present in the source text.

Source Text:
{{TEXT}}


### V1

You are an expert summarizer.

Task:
Summarize the following text in one short paragraph.

Source Text:
{{TEXT}}


Preferred Prompt Version
V2

V2 is selected as the preferred prompt version for this exercise.

### Reason

V2 maintained the same validation success rate as V1 while providing:

Lower average latency
Lower average token usage
More concise summaries
Successful validation across all executed fixed cases

The selection is based on the current objective of producing concise and efficient summaries while maintaining successful validation.