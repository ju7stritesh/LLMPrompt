from PIL import Image
import requests
import numpy as np

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "bus.jpg"
image = Image.open(url)
classes = ["cat", "dog", "human"]
inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
probs = probs[0].detach().numpy()
max_prob_index = np.argmax(probs)
print ("{} is the class with maximum probability of {}%".format(classes[max_prob_index], str(round(probs[max_prob_index]*100, 2))))
for t in range(len(probs)):
    print (classes[t] + ': ' + str(probs[t]))
