# Phrase-detection
import nltk
import csv
import pandas as pd
import numpy as np
# Reading Training file.
data = pd.read_csv("training_data.tsv", delimiter = '\t', encoding = 'utf-8')
len(data)
# Checking out any random sentence from the training data-set
sentence = data['sent'][27]
sentence

from textblob import TextBlob

# Textblob has ready-made function to extract noun-phrases.
print ("TEXTBLOB")
blob = TextBlob(sentence)

for np in blob.noun_phrases:
    print (np)
    
# Tokenizing sentence into individual words
tokens = nltk.word_tokenize(sentence)
tokens

# for each word a tag is assinged
tagged = nltk.pos_tag(tokens)
tagged

# Nouns and Verb type words can be considered as "important words"
Imp_words = [w[0] for w in tagged if w[1].startswith('N') or w[1].startswith('V')]
Imp_words

from nltk.util import ngrams

n=5
for i in range(1,n+1):
    output = list(ngrams(tokens, i))
    print (output,"\n")
    
# Function which takes a sentence as an input and returns phrase

def calculate(sentence):
    
    words = nltk.word_tokenize(sentence)
    nltk.pos_tag(words)
    
    # defining a chunk grammar,indicating how sentences should be chunked. 
    
    grammar = "NP: {<VB.*>?<RB>?<PRP.*>?<IN>?<DT>?<JJ.*>*<NN.*>+}"
    
    # we create a chunk parser
    parser = nltk.RegexpParser(grammar)
    
    # Test it 
    t = parser.parse(nltk.pos_tag(words))
    
    # Result is a tree 
    a = [s for s in t.subtrees() if s.label() == "NP"]
    
    c = []
    num = []
    
    # don't consider here as it is not there in training
    key  = ["monday","tuesday", "wednesday", "thursday","friday","saturday","sunday","today","tomorrow","yesterday", "reminder", "remind", "th", "pm","am"]
    
    for i in range(len(a)):
        count=0
        phrase = ""
        for j in range(len(a[i])):
            if a[i][j][0].lower() in key:
                phrase = phrase
            else :
                phrase = phrase + str(a[i][j][0]) + " "
                count = count+1
        c.append(phrase)
        num.append(count)
        #print (c)
        #print (num)
       
    
    if(c==[] or max(num)<=1):
        return "Not Found"
    else :
        maxi = max(num)
        for i in range(len(num)):
            if(num[i]==maxi):
                return c[i].rstrip()
                
 
print(sentence,"\n") 
print("Phrase  :   ", calculate(sentence))


# Reading file line by line 
with open("eval_data.txt", 'r+') as f:
    lines = [line.rstrip('\n') for linein f]
    
print (lines[67])

### Create new file 'result.txt' to store result

with open('result.csv', mode='w', newline='') as csv_file:
    fieldnames = ['sent', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(len(lines)):
        writer.writerow({'sent':lines[i],'label': calculate(lines[i])})
# Check Accuracy
with open('result.csv', mode='w', newline='', encoding = 'utf-8') as csv_file:
    fieldnames = ['sent', 'Given_label', 'Predicted_label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    count = 0
    for i in range(len(data)):
        writer.writerow({'sent':data['sent'][i], 'Given_label':data['label'][i], 'Predicted_label':calculate(str(data['sent'][i]))})
        
        if str(data['label'][i]) == calculate(str(data['sent'][i])):
            count = count+1
            
print ("Accuracy : ", (count/len(data))*100, "%")
   
