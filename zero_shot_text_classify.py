from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
sequence_to_classify = "The fact is that I like her"
candidate_labels = ['fact', 'opinion', 'emotion']
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
