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


#Get max coherence score to get perfect lda model. Get min perplexity to get perfect lda model.
maxCoherence = 0
numTopics = 0
minPerplexity = 0
for t in range(2, 21):
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=t, id2word=dictionary, passes=20, workers=3)
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=moviesCorpus, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    perplexity = lda_model.log_perplexity(bow_corpus)
    if coherence_score > maxCoherence:
        maxCoherence = coherence_score
        numTopics = t
        minPerplexity = perplexity
    print('Topic: ',t,' Coherence Score: ',coherence_score,' Perplexity: ', perplexity)
    

#Train bow_corpus.
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=numTopics, id2word=dictionary, passes=20, workers=3)


from pprint import pprint


#For each topic, print top 10 significant terms.
pprint(lda_model.print_topics())


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.show(LDAvis_prepared)
'''
Wait, what am I looking at again? There are a lot of moving parts in the visualization. Here's a brief summary:

On the left, there is a plot of the "distance" between all of the topics (labeled as the Intertopic Distance Map). The plot is rendered in two dimensions according a multidimensional scaling (MDS) algorithm.
Topics that are generally similar should be appear close together on the plot, while dissimilar topics should appear far apart.

The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus.
An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the "selected topic" box in the upper-left.

On the right, there is a bar chart showing top terms.
When no topic is selected in the plot on the left, the bar chart shows the top most "salient" terms in the corpus.
A term's saliency is a measure of both how frequent the term is in the corpus and how "distinctive" it is in distinguishing between different topics.
When a particular topic is selected, the bar chart changes to show the top most "relevant" terms for the selected topic.

The relevance metric is controlled by the parameter λλ, which can be adjusted with a slider above the bar chart.

Setting the λλ parameter close to 1.0 (the default) will rank the terms solely according to their probability within the topic.
Setting λλ close to 0.0 will rank the terms solely according to their "distinctiveness" or "exclusivity" within the topic — i.e., terms that occur only in this topic, and do not occur in other topics.
Setting λλ to values between 0.0 and 1.0 will result in an intermediate ranking, weighting term probability and exclusivity accordingly.

Rolling the mouse over a term in the bar chart on the right will cause the topic circles to resize in the plot on the left, to show the strength of the relationship between the topics and the selected term.

Unfortunately, though the data used by gensim and pyLDAvis are the same, they don't use the same ID numbers for topics. If you need to match up topics in gensim's LdaMulticore object and pyLDAvis' visualization, you have to dig through the terms manually.
'''


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


docMostRelevantTopic = np.argmax(docTermMatrix, axis=1) #ith index stores the index of the most relevant topic in ith document.


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
userRateFreq = np.zeros((numUsers, numTopics))


#Load and get dataset.
col2 = dataextract['movieId']
col3 = dataextract['rating']
for i in range(len(dataextract)):
    j = usersIndexMapping[col1[i]]
    k = moviesIndexMapping[col2[i]]
    docMostRelevantTopicIndex = docMostRelevantTopic[k]
    if col3[i]>=3:
        userTermMatrix[j,docMostRelevantTopicIndex] += docTermMatrix[k,docMostRelevantTopicIndex]
    else:
        userTermMatrix[j,docMostRelevantTopicIndex] -= docTermMatrix[k,docMostRelevantTopicIndex]
    userRateFreq[j,docMostRelevantTopicIndex] += 1
userRateFreq[userRateFreq == 0] = 1
for i in range(numUsers):
    userTermMatrix[i] /= userRateFreq[i]


#View userTermMatrix and docTermMatrix
file1 = open("test1.txt", "w")
for i in range(numUsers):
    file1.write(str(userTermMatrix[i]))
    file1.write("\n")
file1.close()
file2 = open("test2.txt", "w")
for i in range(numMovies):
    file2.write(str(docTermMatrix[i]))
    file2.write("\n")
file2.close()


from scipy.stats import pearsonr as pearsons_correlation


#Check and get accurracy. 
file = open("test3.txt", "w")
for i in range(len(dataextract)):
    uid = col1[i]
    mid = col2[i]
    rval = col3[i]
    j = usersIndexMapping[uid]
    uservec = userTermMatrix[j]
    k = moviesIndexMapping[mid]
    docvec = docTermMatrix[k]
    docMostRelevantTopicIndex = docMostRelevantTopic[k]
    #coeff, pval = pearsons_correlation(docvec, uservec) #Pearson's correlation similarity
    #coeff = np.linalg.norm(docvec-uservec) #Euclidean distance similarity
    coeff = abs(docvec[docMostRelevantTopicIndex]-uservec[docMostRelevantTopicIndex]) #Euclidean distance between only relevant topic.
    string = str(uid)+"\t"+str(mid)+"\t"+str(rval)+"\t"+str(coeff)+"\n"
    file.write(string)
file.close()


import operator


def compare(x):
    return x[1]
def run():
    uid = int(input("Enter User Id: "))
    if uid not in usersIndexMapping.keys():
        print("User Id not in record.")
        return
    i = usersIndexMapping[uid]
    uservec = userTermMatrix[i]
    #print(uservec)
    recFactor = []
    for i in range(numMovies):
        docvec = docTermMatrix[i]
        docMostRelevantTopicIndex = docMostRelevantTopic[i]
        #coeff, pval = pearsons_correlation(docvec, uservec) #Pearson's correlation similarity
        #coeff = np.linalg.norm(docvec-uservec) #Euclidean distance similarity
        coeff = abs(docvec[docMostRelevantTopicIndex]-uservec[docMostRelevantTopicIndex]) #Euclidean distance between only relevant topic.
        #print(str(moviesName[i])+" "+str(moviesId[i]))
        recFactor.append(tuple((i, coeff)))
    recFactor = sorted(recFactor, key=operator.itemgetter(1), reverse=True)
    numRec = 10
    recommend = []
    for j in range(numRec):
        i = recFactor[j][0]
        recommend.append(tuple((moviesName[i], moviesId[i])))
    print(recommend)

if __name__=="__main__":
    run()