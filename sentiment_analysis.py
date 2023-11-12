from transformers import pipeline
import text2emotion as te
# from nrclex import NRCLex

# create pipeline for sentiment analysis
classification = pipeline('sentiment-analysis')

a = "You guys should see their real life, they are the best"
b = "I did not understand anything in this movie."
c = "UP! was sold out, so i'm seeing Night At The Museum 2. I'm 12 years old."
d = "I am gonna kil you"

a_res = classification([a,b,c,d])

emot = te.get_emotion(b)

print (a_res)
print (emot)

# Install emoji==1.7 library

