import nltk
import random
import requests
import spacy
from nltk.corpus import wordnet
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

nlp = spacy.load("en_core_web_sm")
nltokenizer = nltk.tokenize.word_tokenize

# Shuffle the words in a sentence
def shuffle(sentence, task, type):
    # First check whether the task is hendrycks
    if task == "hendrycks":
        return hendrycks_shuffle(sentence, type)
    # Keep track of whether we need to re-append the "Question:" token
    appendQuestion = False
    appendAnswer = False
    words = nltokenizer(sentence) # Tokenize the sentence into words
    # Check if the question contains "Question:" and "Answer:"
    if words[0] == "Question":
        # Remove the tokens 'Question' and ':' for the shuffle
        words = words[2:]
        appendQuestion = True
    if words[-2] == "Answer":
        # Remove the tokens 'Answer' and ':' for the shuffle
        words = words[:-2]
        appendAnswer = True
    if type == "unigram":
        words = unigram_shuffle(words)
    if type == "bigram":
        words = bigram_shuffle(words)
    if type == "trigram":
        words = trigram_shuffle(words)
    # Turn the words back into a string
    words = " ".join([word for word in words])
    # Re-append the "Question:" token if it was removed
    if appendQuestion:
        words = "Question: " + words
    if appendAnswer:
        words = words + " Answer:"
    return words  # Join the shuffled words back into a sentence
    
# Shuffle sets of one word
def unigram_shuffle(words):
    random.shuffle(words)  # Shuffle the order of words
    return words

# Shuffle sets of two words
def bigram_shuffle(words):
    bigrams = list(nltk.bigrams(words))  # Create bigrams
    random.shuffle(bigrams)  # Shuffle the order of bigrams
    shuffled_words = [word for bigram in bigrams for word in bigram]  # Flatten back to words
    return shuffled_words

# Shuffle sets of three words
def trigram_shuffle(words):
    trigrams = list(nltk.trigrams(words))  # Create trigrams
    random.shuffle(trigrams)  # Shuffle the order of trigrams
    shuffled_words = [word for trigram in trigrams for word in trigram]  # Flatten back to words
    return shuffled_words

# In Hendrycks the answers are listed in the prompt. To shuffle the question, find everything before "A."
def hendrycks_shuffle(sentence, type):
    # Find the index of "A" in the sentence
    index_of_A = sentence.find("A.")
    words_before_A = sentence[:index_of_A]
    words_after_A = sentence[index_of_A:]
    words_before_A = nlp(words_before_A)
    # Shuffle the words before "A"
    if type == "unigram":
        random.shuffle(words_before_A)
    elif type == "bigram":
        words_before_A = bigram_shuffle(words_before_A)
    elif type == "trigram":
        words_before_A = trigram_shuffle(words_before_A)
    # Combine the shuffled and unshuffled parts of the sentence
    shuffled_words = ' '.join(words_before_A) + '\n' + words_after_A
    return shuffled_words

# Replace verbs with antonyms, May extend this to accept a POS as an argument
def getSynonym(word):
    synsets = wordnet.synsets(word, pos=wordnet.VERB)
    synonyms = []
    # Check if there are any synsets available
    if synsets:
        # Get the first synset
        key = synsets[0]
        # Get synonyms
        synonyms = [str(lemma.name()) for lemma in key.lemmas()]
        # Get hypernyms and add them to the list
        for hypernym in key.hypernyms():
            synonyms.extend([str(lemma.name()) for lemma in hypernym.lemmas()])
    return synonyms

# Extract verbs from the prompt and replace them
def verbSynonyms(words):
    doc = nlp(words)
    text = []
    for token in doc:
        if token.pos_ == 'VERB':
            # Get the synonyms
            synonyms = getSynonym(token.text)
            if len(synonyms) > 1:
                text.append(synonyms[1])
            else:
                text.append(token.text)
        else:
            text.append(token.text)
    return " ".join(text)

# Gets the subject of a sentence. If one is not found, will return first noun, then first word
def get_sentence_subject(sentence):
    doc = nlp(sentence)
    # Get the first named entity in the sentence using spacy
    for ent in doc.ents:
        return ent.text, 'NE'
    # If there is no named entity, try and get the subject
    for token in doc:
        if token.dep_ == 'nsubj':
            return token.text, 'SUB'  
    return doc[0].text, 'ART'

# Load the model and tokenizer
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
    # Print teh model device
    print("Model Device:", next(model.parameters()).device)
    return tokenizer, model

# Generate a fake answer using a model
def generate_fake_answer(word, pos, model, tokenizer):   
    # Use the pipeline to generate text
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Start generating a sentence with the word
    if pos == 'NE' or pos == 'SUB': 
        prompt = "The " + word
    
    # If no nouns are found the sentence will be completely random
    else:
        prompt = word

    # Generate text
    answer = generator(prompt, max_length=30)
    answer = answer[0]['generated_text']
    answer = answer.replace('<200b>', '')  # Remove <200b> characters
    if '.' in answer:
        answer = answer[:answer.index('.') + 1]
    return answer

# Remove the named entities from the sentence
def removeNE(sentence):
    # Get the named entities in the sentence using spacy
    doc = nlp(sentence)
    for ent in doc.ents:
        sentence = sentence.replace(ent.text, '')
    return sentence

# Keep only the named entities in the sentence
def onlyNE(sentence):
    # Get the named entities in the sentence using spacy
    doc = nlp(sentence)
    named_entities = [ent.text for ent in doc.ents]
    return ' '.join(named_entities)


