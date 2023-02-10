import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print('\n')


tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
print('\n')

tokens = nlp('horses perfume bees nectar')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
print('\n')

# Write a note about what you found interesting about the similarities between cat, monkey and banana and think of an example of your own
''' similarities between cat, monkey and banana '''
# interesting to know there's more similarity between a cat and a monkey than 
# there is between the monkey and a banana

''' my example horses, perfume, bees and nectar '''
# almost a true comparison between the bees and nectar and
# horses and perfume as in reality 
# horses don't take well to strong scents and bees feed on nectar


sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
            "Hello, there is my car",
            "I\'ve lost my car in my car",
            "I\'d like my boat back",
            "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)



''' Run the example file with the simpler language model 'en_core_web_sm' and
write a note on what you notice is different from the model 'en_core_web_md' '''
# the language model 'en_core_web_sm' splits up the data with almost a fine tooth-comb and even recognises punctuation
# but on the other hand it is a small model and does not give useful similarity judgements as it has no word vectors loaded
# while language model 'en_core_web_md' is a larger model so can do both the data spliting as the 'en_core_web_sm' and
# give useful similarity judgements as well
