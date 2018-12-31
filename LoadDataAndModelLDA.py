from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


stopwordsList1 = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 
'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 
'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 
'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 
'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 
'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 
'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 
'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 
'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 
'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 
'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 
'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 
'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 
'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 
'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 
'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 
'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 
'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 
'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 
'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 
'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer',
'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 
'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 
'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer',
'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 
'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on',
'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 
'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 
'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point',
'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 
'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 
'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 
'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 
'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 
'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 
'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 
'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 
'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 
'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 
'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 
'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 
'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 
'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 
'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 
'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 
'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 
'yours', 'z']
stopwordsList2 = list(stopwords.words('english'))
lemma = WordNetLemmatizer()    
stemma = PorterStemmer()


import pandas as pd


moviesName = [] #To be returned.
moviesId = [] #To be returned.
numMovies = 0 #To be returned.


moviesIndexMapping = {}
moviesCorpus = []


#Load and get dataset.
dataextract = pd.read_csv('movies.csv')
numMovies = len(dataextract)
col1 = dataextract['movieId']
col2 = dataextract['title']
col3 = dataextract['genres']
for i in range(numMovies):
    moviesName.append(col2[i])
    moviesId.append(col1[i])
    doc = []
    wordsList = col3[i].split('|')
    for j in range(len(wordsList)):
        word = wordsList[j]
        word = word.lower()
        word = stemma.stem(lemma.lemmatize(word))
        if word in stopwordsList1:
            continue;
        elif word in stopwordsList2:
            continue;
        else:
            doc.append(word)
    moviesCorpus.append(doc)
    moviesIndexMapping[col1[i]] = i


#Load and get dataset.
dataextract = pd.read_csv('tags.csv')
col1 = dataextract['movieId']
col2 = dataextract['tag']
for i in range(len(dataextract)):
    word = col2[i]
    word = word.lower()
    word = stemma.stem(lemma.lemmatize(word))
    if word in stopwordsList1:
        continue;
    elif word in stopwordsList2:
        continue;
    else:
        j = moviesIndexMapping[col1[i]]
        moviesCorpus[j].append(word)
        

import gensim


#Bag of words. Create a dictionary containing the word and words unique ID.
dictionary = gensim.corpora.Dictionary(moviesCorpus)
dictionary.filter_extremes(no_below=5) #Alter according to dataset.


#For each document a list of tuples is created reporting the words(ID) in filtered dictionary and frequency of those words.
bow_corpus = [dictionary.doc2bow(doc) for doc in moviesCorpus]


#Get lda model. Train the model over bow_corpus.
numTopics = 20
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=numTopics, id2word=dictionary, passes=3, workers=2)
#For each topic, explore the words occuring in that topic and its relative weight in documents.
for indx in range(numTopics):
    print("Topic: "+str(indx))
    print(lda_model.print_topics(indx))
    print("\n")


import numpy as np


docTermMatrix = np.zeros((numMovies, numTopics)) #To be returned.


#Fill document term matrix
for i in range(len(moviesCorpus)):
    doc = moviesCorpus[i]
    bow_vector = dictionary.doc2bow(doc)
    vec = lda_model[bow_vector]
    #print(doc)
    #print(vec)
    for j in range(len(vec)):
        t = vec[j]
        docTermMatrix[i,j] = t[1];


numUsers = 0 #To be returned.
usersIndexMapping = {} #To be returned.


#Count number of users.
dataextract = pd.read_csv('ratings.csv')
col1 = dataextract['userId']
for i in range(len(dataextract)):
    if col1[i] not in usersIndexMapping:
        usersIndexMapping[col1[i]] = numUsers
        numUsers += 1


userTermMatrix = np.zeros((numUsers, numTopics)) #To be returned.


userRatingFreq = {}


#Load and get dataset.
col2 = dataextract['movieId']
col3 = dataextract['rating']
for i in range(len(dataextract)):
    if col3[i]<3:
        continue
    if col1[i] in userRatingFreq:
        userRatingFreq[col1[i]] += 1
    else:
        userRatingFreq[col1[i]] = 1
    j = usersIndexMapping[col1[i]]
    k = moviesIndexMapping[col2[i]]
    userTermMatrix[j] += (col3[i]-3)*docTermMatrix[k]
for key, val in userRatingFreq.items():
    j = usersIndexMapping[key]
    userTermMatrix[j] /= val


from scipy.stats import pearsonr as pearsons_correlation


def run():
    #uid = int(input("Enter User Id: "))
    for j in range(numUsers):
        uid = j+1
        if uid not in usersIndexMapping.keys():
            continue;
        i = usersIndexMapping[uid]
        uservec = userTermMatrix[i]
        print(uid)
        print(uservec)
        numRec = 0
        for i in range(numMovies):
            coeff, pval = pearsons_correlation(docTermMatrix[i], uservec)
            if coeff>0.9:
                print(str(moviesName[i])+" "+str(moviesId[i]))
                numRec += 1
        print(numRec)


if __name__=="__main__":
    run()