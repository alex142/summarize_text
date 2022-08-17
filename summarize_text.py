from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
from collections import defaultdict

article_url = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/"

def getTextFromUrl(url : str) -> str:
    #download the page HTML using urllib
    page = urlopen(article_url).read().decode('utf8', 'ignore')
    #parse text by tag using beautiful_soup
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    #encode and return
    return text


def get_word_distribution(text : str, sents : list) -> FreqDist:
    word_sents = word_tokenize(text.lower())
    custom_stop_words = set(stopwords.words("english") + list(punctuation) + ['“', '”'])
    words_wo_stopwords = [word for word in word_sents if word not in custom_stop_words]
    return  FreqDist(words_wo_stopwords)

def get_absract(text : str, n : int) -> str:
    #init ranking dict from collections lib
    ranking = defaultdict(int)
    #get sentences list
    sents = sent_tokenize(text)

    assert len(sents) >= n
    dist = get_word_distribution(text, sents)

    #do sentence ranking by word occurence (how many top ranked words in each sentence)
    #enumerate() gives list of tuples (index, item)
    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in dist:
                ranking[i] += dist[w]

    #get top N ranked sentences with heapq, sort and join
    sents_index = nlargest(n, ranking, key=ranking.get)
    abstract = [sents[j] for j in sorted(sents_index)]
    return ' \b '.join(abstract)

text = getTextFromUrl(article_url)

print(get_absract(text, 4))