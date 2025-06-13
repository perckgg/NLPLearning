import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Display POS tags
for token in doc:
    print((token.text, token.pos_, token.tag_))
