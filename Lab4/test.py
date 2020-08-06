# %%
import nltk

sentence = "A vibrant, young city, Calgary is home to a thriving innovation sector."

# %%
nltk.download('punkt')
tokens = nltk.word_tokenize(sentence)
print(tokens)

# %%
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print([t for t in tokens if t.lower() not in stop_words])

# %%
