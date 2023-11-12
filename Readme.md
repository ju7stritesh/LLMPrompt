This repo is created to use the latest in AI research that is readily available from Hugging face's transformer library and other sources using just a few lines of code.
The models are chosen based on the ease of use and their performance. There may be better models and results but those may require multiple GPUs and the results are sometimes only slightly better the ones below.

#Image classification using zero shot detection

Filename - image_prompt_zero_sort.py
The CLIP model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision tasks. 
The model was also developed to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. 
It was not developed for general model deployment - to deploy models like CLIP, 
researchers will first need to carefully study their capabilities in relation to the specific context they’re being deployed within.
The base model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. 
These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.
Image ![bus.jpg](bus.jpg)

Input classes [dog, cat, human]
Output - cat: 0.025230074
dog: 0.011377574
human: 0.9633924

Input classes [bus, dog, cat, human]
Output - bus: 0.99950945
cat: 1.2375028e-05
dog: 5.580544e-06
human: 0.00047253008


Limitations:
The likelihood of human detection decreases significantly when the bus is included, as demonstrated above.

Filename - image_gen.py
The Segmind Stable Diffusion Model (SSD-1B) is a distilled 50% smaller version of the Stable Diffusion XL (SDXL), 
offering a 60% speedup while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, 
including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.
This model employs a knowledge distillation strategy, where it leverages the teachings of several expert models in succession, including SDXL, ZavyChromaXL, 
and JuggernautXL, to combine their strengths and produce impressive visual outputs.

_Note : We can reduce the image size to run on smaller GPUs and reduce the number of inference steps for quicker results_

Input : prompt = "A cute cat"
neg_prompt = "ugly, blurry, poor quality"
 Output:![cat.jpg](cat.jpg)

Limitations : Realism in human depictions, larger GPU requirement for better results, understanding of input prompt.

Filename - semantic_similarity.py
This is a sentence-transformers model: 
It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

Input: ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

Output: The new movie is awesome 		 The new movie is so great 		 Score: 0.8939
The cat sits outside 		 The cat plays in the garden 		 Score: 0.6788
I love pasta 		 Do you like pizza? 		 Score: 0.5096
I love pasta 		 The new movie is so great 		 Score: 0.2560
I love pasta 		 The new movie is awesome 		 Score: 0.2440
A man is playing guitar 		 The cat plays in the garden 		 Score: 0.2105
The new movie is awesome 		 Do you like pizza? 		 Score: 0.1969
The new movie is so great 		 Do you like pizza? 		 Score: 0.1692
The cat sits outside 		 A woman watches TV 		 Score: 0.1310
The cat plays in the garden 		 Do you like pizza? 		 Score: 0.0900

Filename - image_to_text.py
The VisionEncoderDecoderModel can be used to initialize an image-to-text model with any pretrained Transformer-based vision model as the encoder (e.g. ViT, BEiT, DeiT, Swin) and any pretrained language model as the decoder (e.g. RoBERTa, GPT2, BERT, DistilBERT).

The effectiveness of initializing image-to-text-sequence models with pretrained checkpoints has been shown in
Input : ![bus.jpg](bus.jpg)
Output ['a bus stops at a bus stop to pick up passengers']

Filename : sentiment_analysis.py
Label text according to some polarity like positive or negative which can inform and support decision-making in fields like politics, finance, and marketing
Emotions can be analyzed using an open source library like text2emotion

Input : a = "You guys should see their real life, they are the best"
b = "I did not understand anything in this movie."
c = "UP! was sold out, so i'm seeing Night At The Museum 2. I'm 12 years old."
d = "I am gonna kil you"

Output : [{'label': 'POSITIVE', 'score': 0.9998785257339478}, {'label': 'NEGATIVE', 'score': 0.9994907379150391}, {'label': 'POSITIVE', 'score': 0.9986440539360046}, {'label': 'POSITIVE', 'score': 0.9968469738960266}]
{'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 1.0, 'Fear': 0.0}

Filename : zero_shot_text_classify.py

This model was trained on the MultiNLI, Fever-NLI and Adversarial-NLI (ANLI) datasets, which comprise 763 913 NLI hypothesis-premise pairs. This base model outperforms almost all large models on the ANLI benchmark. The base model is DeBERTa-v3-base from Microsoft. The v3 variant of DeBERTa substantially outperforms previous versions of the model by including a different pre-training objective

Input : sequence_to_classify = "The fact is that I like her"
candidate_labels = ['fact', 'opinion', 'emotion']

output : {'sequence': 'The fact is that I like her', 'labels': ['opinion', 'emotion', 'fact'], 'scores': [0.40460214018821716, 0.34235116839408875, 0.2530467212200165]}

