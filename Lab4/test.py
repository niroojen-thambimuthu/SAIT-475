# %%
# # import nltk
# # from nltk.stem.porter import PorterStemmer

# # stemmer = PorterStemmer()
# # words = ['compute', 'computers', 'computed', 'computing']

# # for w in words:
# #     print(f"{w} -> {stemmer.stem(w)}")

# # %%
# import nltk
# nltk.download('punkt')

# sentence = """
# When the Boogeyman goes to sleep every night,
# he checks his closet for Chuck Norris.
# """


# # %%
# nltk.download('stopwords')
# tokens = nltk.word_tokenize(sentence)
# print(tokens)

# # %%
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# print([t for t in tokens if t.lower() not in stop_words])

# # %%
# nltk.download('averaged_perceptron_tagger')
# tagged = nltk.pos_tag(tokens)
# print(tagged)


# # %%
# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for token in doc:
#     print(token.text)


# # %%
# for token in doc:
#     print(
#         token.text, token.lemma_, token.pos_, token.tag_,
#         token.dep_, token.shape_, token.is_alpha, token.is_stop
#     )

# %%
# from gensim.models import Word2Vec
# sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
# model = Word2Vec(sentences, min_count=1)
# print(model.wv.similarity("cat", "dog"))
# print(model.wv.similarity("meow", "woof"))

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

data, _ = fetch_20newsgroups(
    shuffle=True, random_state=1,
    remove=('headers', 'footers', 'quotes'),
    return_X_y=True
)

tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2,
    max_features=1000, stop_words='english'
)

tf = tf_vectorizer.fit_transform(data)

lda = LatentDirichletAllocation(
    n_components=10,
    max_iter=5,
    learning_method='online',
    learning_offset=50,
    random_state=0
)
lda.fit(tf)


# %%
def show_topic(model, feature_names, top):
    for index, distribution in enumerate(model.components_):
        sorted_word_indices = distribution.argsort()[::-1][:top]
        print(f"Topic {index}:")
        feats = [feature_names[i] for i in sorted_word_indices]
        print(" ".join(feats))

tf_feature_names = tf_vectorizer.get_feature_names()
show_topic(lda, tf_feature_names, 10)

# %%
