# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:54:00 2020

@author: eilxaix
# """
import json
import re
import nltk
from nltk.tokenize import sent_tokenize

import spacy
# # Load the spacy model that you have installed
# # nlp = spacy.load("en")
scinlp = spacy.load("en_core_sci_sm")


# # from scispacy.abbreviation import AbbreviationDetector
# # abbreviation_pipe = AbbreviationDetector(nlp)
# # scinlp.add_pipe(abbreviation_pipe)


# # import string
# # punct = string.punctuation

# # # =============================================================================
from nltk import word_tokenize
STOPWORDS = ["a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'just', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'que', 'quite', 'qv', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'value', 'various', 'very', 'via', 'viz', 'vs', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', "wouldn't", 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'zero']
stopwords = STOPWORDS
stop_words = set(stopwords.words("english"))
# # # =============================================================================
# # from constants import STOPWORDS
# # # stop_words=nltk_stopwords
# # stop_words=STOPWORDS

# # from nltk.stem.wordnet import WordNetLemmatizer
# # from nltk.stem.porter import PorterStemmer
# # lem = WordNetLemmatizer()
# # ps=PorterStemmer()

# # POS_BLACKLIST = ['INTJ', 'AUX', 'CCONJ', 'ADP', 'DET', 'NUM', 'PART','PRON', 'SCONJ', 'PUNCT','SYM', 'X']

# # from collections import Counter, defaultdict


# def lemmatize(chunk):
#     tokens = []
#     if type(chunk) == spacy.tokens.span.Span:
#         for token in [chunk[0], chunk[-1]]:
#             if token.pos_ not in POS_BLACKLIST:
#                 tokens.append(token.text)
#         text = ' '.join(tokens).lower()
#     else:
#         text = chunk
    
#     #Remove punctuations
#     text = re.sub('[^a-zA-Z0-9]', ' ', text)

#     #remove tags
#     text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

#     # remove special characters and digits
# #     text=re.sub("(\\d|\\W)+"," ",text)

#     ##Convert to list from string
#     text = text.split()

#     #Lemmatisation
#     lem = WordNetLemmatizer()
#     text = [lem.lemmatize(word) for word in text if not word in  
#             stop_words] 
#     text = " ".join(text)
#     return text

# def preprocess(chunk):
#     tokens = []
#     if type(chunk) == spacy.tokens.span.Span:
#         for token in [chunk[0], chunk[-1]]:
#             if token.pos_ not in POS_BLACKLIST:
#                 tokens.append(token.text)
#         text = ' '.join(tokens).lower()
#     else:
#         text = chunk
#     text = text.lower()
#     text = re.sub('[^A-Za-z0-9]+', ' ', text)
#     token = re.split(' ',text)
#     text = [tok for tok in token if not tok in stop_words and not tok in punct] 
#     text = " ".join(text)
#     return text    

# def np_extraction(text):
#     nounphrases = []
#     doc = nlp(text)
#     for chunk in doc.noun_chunks:
#         candidate = preprocess(chunk.text)
#         if candidate != '':
# #             all_text[i] = all_text[i].replace(chunk.text, text)
#             nounphrases.append(candidate)
#     return nounphrases

# def find_subset_np(nounphrases):
#     subset = []
#     for np in nounphrases:
#         length = len(np.split())
#         if length > 4:
#             subnp = np_extraction(np)
#             subset += [i for i in subnp if i not in nounphrases]
#     return subset

        
# # =============================================================================
# # from stanfordcorenlp import StanfordCoreNLP
# # en_model = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05',quiet=True)
# # =============================================================================

# def pos_tag_standford(text, en_model,stop_words):
#     tokens = en_model.word_tokenize(text)
#     tokens_tagged = en_model.pos_tag(text)
#     for i, token in enumerate(tokens):
#         if token.lower() in stop_words:
#             tokens_tagged[i] = (token, "IN")
#     return tokens, tokens_tagged

def pos_tag_spacy(text, nlp, stop_words):

    doc = nlp(text)
    tokens = []
    tags = []
    tokens_tagged = []
    for tok in doc:
        tokens.append(tok.text)
        tokens_tagged.append((tok.text,tok.tag_))
        tags.append('<%s>' % tok.tag_ )
    return tokens, tags, tokens_tagged

    
def extract_candidates(tokens, tokens_tagged, no_subset=True):

    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    GRAMMAR1 = """  NP:
        {<NN.*|JJ|CD>*<NN.*>}   # Adjective(s)(optional) + Noun(s)"""
       
    GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""
    
    GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN|CD>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
     
    GRAMMAR = """
            NP:
              {<NN.*|JJ|CD>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            VBNP:
              {<VBG|VBN>*<NN.*>} # Above, connected with in/of/etc...
            """

    for i, token in enumerate(tokens):
        if token.lower() in stop_words:
            tokens_tagged[i] = (token, "IN")
    
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = {}
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves()).lower()
            length = len(token.leaves())
            if np in keyphrase_candidate:
                keyphrase_candidate[np].append((count, count + length)) 
            else:
                keyphrase_candidate[np] = [(count, count + length)]
            count += length

        else:
            count += 1

    return keyphrase_candidate


def find_acronyms(text):
    doc = scinlp(text)
    abrv_list = []
    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        abrv_list.append((abrv.text, str(abrv._.long_form)))

    return list(set(abrv_list))

# def replace_acronyms(text):
#     doc = nlp(text)
#     altered_tok = [tok.text for tok in doc]
#     for abrv in doc._.abbreviations:
#         altered_tok[abrv.start] = str(abrv._.long_form)

#     return(" ".join(altered_tok))

# def filter_unigram(nps, abrv):
#     nps_validate = []
#     for np in nps:
#         length = len(np.split())
#         if length > 1 or np in abrv:
#             nps_validate.append(np)
#     return nps_validate

# def no_punct(word):
#     for tok in word.split():
#         if re.match('\W+', tok):
#             return False
#     return True 

# def filtered_pos(words, words_tags_dict):
#     pattern = re.compile('^(<JJ>|<NN>)*(<NN>|<NNS>|<NNP>)+$')
#     filtered = []
#     for w in words:
#         tags = words_tags_dict[w]
#         match = True
#         for tag in tags:
#             if pattern.match(tag) is None:
#                 match = False
#                 break
#         if match == True:
#             filtered.append(w)
#     return filtered           
 
# def n_gram_words(tokens, tags, n_gram = 8):
#     """
#     To get n_gram word frequency dict
#     input: tokens list,int of n_gram 
#     output: word frequency dict

#     """
#     words = []
#     words_tags = []
#     words_tags_dict = defaultdict(list)
#     for i in range(1,n_gram+1):
#         words += [' '.join(tokens[j:j+i]) for j in range(len(tokens)-i+1)]
#         words_tags += [''.join(tags[j:j+i]) for j in range(len(tags)-i+1)]
#     for w in range(len(words)):
#         words_tags_dict[words[w]].append(words_tags[w])
# # =============================================================================
# #     words_freq = dict(Counter(words))    
# # =============================================================================
#     words = [w for w in filtered_pos(words, words_tags_dict) if no_punct(w)]
#     return list(set(words)) 

def candidates_extraction(data):
    # input: [data] - a list of sentence
    # output: candidates:{'nps':[], 'sub_nps':[], 'abrv':[]}
    candidates={}
    abrv_corpus = {}
    for i,text in enumerate(data):
        # find_acronyms
        abrv_list = find_acronyms(text)  
        abrv_corpus[i]=abrv_list
        abrv = [i[0] for i in abrv_list]
        
        # pos tagging
        tokens, tags, tokens_tagged = pos_tag_spacy(text, nlp, stop_words)
        # ngram + pos matching '(<JJ>)*(<NN>|<NNS>|<NNP>)'
        ngrams = n_gram_words(tokens, tags, n_gram = 8)

        # np chunking
        # nps = np_extraction(text)
        nps = extract_candidates(tokens, tokens_tagged)

        candidates[i] = {'nps':nps, 'ngrams': ngrams, 'abrv':abrv}
    return candidates, abrv_corpus
    
def acronym_extraction(data):
    # input: [data] - a list of abstract
    # output: abrv：{'id':[]}, abrv_corpus: {'ai': ..., }
    abrv={}
    for i,text in enumerate(data):
        # find_acronyms
        abrv_list = find_acronyms(text)  
        abrv[i] = [[i[0],i[1]] for i in abrv_list]
    return abrv


if __name__ == "__main__":
    file_path = "/home/jazhan/code/document2slides-main/d2s-model/0.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for section_text in data["text"]:
            cur_section = section_text["section"]
            n = section_text['n']   # 章节编号
            content_string = section_text['strings']
            content_string = sent_tokenize(content_string)
            candi_keys = candidates_extraction(content_string)
            print("cur_section")
            print(content_string)