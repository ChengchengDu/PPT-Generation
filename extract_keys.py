#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""针对一个json文件关键词抽取"""
from collections import defaultdict
import json
import time
from nltk.util import pr
import numpy as np
import os
from typing import DefaultDict, KeysView
from numpy.core.defchararray import add, endswith, title
from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import break_cycles
from openie import StanfordOpenIE
from stanfordcorenlp import StanfordCoreNLP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from spherecluster import SphericalKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
from pptx import Presentation
import nltk
from nltk import text
import torch
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import sys
sys.path.append('/home/jazhan/code/document2slides-main/d2s_model/')
import create_pptx

os.environ["TOKENIZERS_PARALLELISM"] = "false"

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
stop_words = set(stopwords.words("english"))
wnl=nltk.WordNetLemmatizer()
considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ','VBG'}
#GRAMMAR1 is the general way to extract NPs

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, text="", openie=False):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        text = text.strip()
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.tokens = []
        self.tokens_tagged = []
        self.nn_keyphrase_candidate = []
        self.tokens = nltk.word_tokenize(text)
        self.tokens_tagged = nltk.pos_tag(self.tokens)
        if openie:
            self.trples_candi_keys = get_candidate_triples_keys(text)
        else:
            self.trples_candi_keys = []
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stop_words:
                self.tokens_tagged[i] = (token, "IN")
        np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
        self.np_pos_tag_tokens = np_parser.parse(self.tokens_tagged)
        count = 0
        for token in self.np_pos_tag_tokens:
            if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
                np = ' '.join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_pos = text.find(np)
                start_end = (count, count + length)
                count += length
                self.nn_keyphrase_candidate.append((np, start_end))
            else:
                count += 1

    
def get_candidate_triples_keys(text):
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triples = set()
        subjects = set()
        objects = set()
        candidate_keys = set()
        for triple in client.annotate(text):
            # 关键词候选对象: 头实体 尾实体 和 关系三元组
            subject = triple['subject']
            relation = triple['relation']
            object = triple['object']
            cur_triple = subject + ' ' + relation + ' ' + object
            # 限制关键词组最多5个token
            if len(cur_triple.split(' ')) > 5:
                triples.add(cur_triple)
        
        return list(triples)


def get_core_keys(content, model):
    """抽取最终的关键词  利用语义信息抽取关键词"""
    candidate_keys = get_candidate_triples_keys(content)
    content_embedding = model.encode([content])
    candidate_embedding = model.encode(candidate_keys)
    top_n = 3
    return MMRel(content_embedding, candidate_embedding, candidate_keys, top_n=top_n, diversity=0.3)


def read_json(file_path, model):
    """
    一个data的结构是:
    {
        "title":string,
        "abstract":string,
        "text", [{"section_name": string, "n":"2.1", "string":"vsgfdb"}, {}],
        "headers": [{"section":section_name, "n": 2, "len": 5}, {}, {}],
        figtures:[{}, {}, {}],
    }


    最终的关键词的结构是：
    {
        "title":string,
        "abstract": [keys1, key2,...],
        "text":["section_name": [keys1, key2,...], "section_name":[]],

    }
    """
    
    keys_dict = {
        "title": "",
        "abstract": [],
        "keywords":[],
    }
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for section_text in data["text"]:
            cur_section = section_text["section"]
            n = section_text['n']   # 章节编号
            content_string = section_text['strings']
            keys = get_core_keys(content_string, model) 
            keys_dict["title"] = data["title"]
            keys_dict["abstract"] = data["abstract"]

            cur_keys = {"section":cur_section, "n": n, "keys": " . ".join(keys)}
            keys_dict["keywords"].append(cur_keys)
    with open("/home/jazhan/code/document2slides-main/d2s-model/keys-0.json", 'w', encoding="utf-8") as f:
        json.dump(keys_dict, f)



def MMRel(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


class SentEmbeddings():
    def __init__(self, 
                embed_model,
                weightfile_pretrain='/home/jazhan/code/document2slides-main/sciduet-build/enwiki_vocab_min200.txt',
                weightfile_finetune='/home/jazhan/code/document2slides-main/input/fre_paper/finetune_fre.txt',
                weightpara_pretrain=2.7e-4,
                weightpara_finetune=2.7e-4,
                lamda=1.0,
                database="paper"
                ):
        "finetune的文件是当前的语料库  pretrained是通用语料库"
        self.embed_model = embed_model
        self.word2weight_pretrain = get_word_weight(weightfile_pretrain, weightpara_pretrain)   # 获取每一个token对应的权重
        self.word2weight_finetune = get_word_weight(weightfile_finetune, weightpara_finetune)
        self.lamda=lamda
        self.database=database

    def get_tokenized_sent_embeddings(self, text_obj):
        tokens_segmented = get_sent_segmented(text_obj.tokens)
        bert_embeddings = self.embed_model.get_tokenized_words_embeddings(tokens_segmented)
        # bert_embeddings:shape(len(tokens), dim)
        bert_embeddings = splice_embeddings(bert_embeddings, tokens_segmented)

        candidate_embedding_list = []

        # 获取每一个token对应的词频
        weight_list = get_weight_list(self.word2weight_pretrain, self.word2weight_finetune, text_obj.tokens, lamda=self.lamda)
        sent_embeddings = get_weighted_average(text_obj.tokens, text_obj.tokens_tagged, weight_list, bert_embeddings)   # bert_embedding[0]表示每一个token的embedding

        nn_candidate_embeddings_list = []
        for kc in text_obj.nn_keyphrase_candidate:
            start = kc[1][0]
            end = kc[1][1]
            kc_emb = get_candidate_weighted_average(text_obj.tokens, weight_list, bert_embeddings, start, end)
            nn_candidate_embeddings_list.append(kc_emb)
    
        return sent_embeddings, nn_candidate_embeddings_list 


def get_candidate_weighted_average(tokenized_sents, weight_list, embeddings_list, start, end):
    # 对候选词组进行加权求和
    # weight_list=get_normalized_weight(weight_list)
    assert len(tokenized_sents) == len(weight_list)
    # num_words = len(tokenized_sents)
    num_words =end - start
    dim = embeddings_list[0].size(0)
    sum_embed = torch.zeros((1, dim))
    for j in range(start, end):
        e_test = embeddings_list[j]
        sum_embed += e_test * weight_list[j]
    sum_embed = sum_embed  / float(num_words)
    return sum_embed

def get_weighted_average(tokenized_sents, sents_tokened_tagged, weight_list, embeddings_list):

    """
    embeddings_list: tensor(len(tokens), dim)
    """
    # weight_list=get_normalized_weight(weight_list)
    # 按照所有token的词频进行加权得到文档的表示
    sum_embed = torch.zeros((1, 768))
    assert len(tokenized_sents) == len(weight_list)
    num_words = len(tokenized_sents)   # 所有token的长度
    for j in range(0, num_words):
        if sents_tokened_tagged[j][1] in considered_tags:
            assert sents_tokened_tagged[j][0] == tokenized_sents[j]
            e_test = embeddings_list[j]
            sum_embed += e_test * weight_list[j]
    return sum_embed


def get_word_weight(weightfile="", weightpara=2.7e-4):
    """
    Get the weight of words by word_fre/sum_fre_words  当前词的词频 / 总的词频
    :param weightfile   词频文件
    :param weightpara
    :return: word2weight[word]=weight : a dict of word weight
    """
    if weightpara <= 0:  # when the parameter makes no sense, use unweighted
        weightpara = 1.0
    word2weight = {}
    word2fre = {}
    with open(weightfile, encoding='UTF-8') as f:
        lines = f.readlines()
    # sum_num_words = 0
    sum_fre_words = 0
    for line in lines:
        word_fre = line.split()
        # sum_num_words += 1
        if (len(word_fre) == 2):
            word2fre[word_fre[0]] = float(word_fre[1])
            sum_fre_words += float(word_fre[1])
        else:
            print(line)
    for key, value in word2fre.items():
        word2weight[key] = weightpara / (weightpara + value / sum_fre_words)
        # word2weight[key] = 1.0 #method of RVA
    return word2weight


def get_weight_list(word2weight_pretrain, word2weight_finetune, tokenized_sents, lamda, database="paper"):
    # 针对每一个词统计词频 tokenized_sents表示tokens
    weight_list = []
    for word in tokenized_sents:
        word = word.lower()

        if(database==""):
            weight_pretrain = get_oov_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight=weight_pretrain
        else:
            weight_pretrain = get_oov_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight_finetune = get_oov_weight(tokenized_sents, word2weight_finetune, word, method="max_weight")
            weight = lamda * weight_pretrain + (1.0 - lamda) * weight_finetune
        weight_list.append(weight)

    return weight_list


def get_oov_weight(tokenized_sents, word2weight, word, method="max_weight"):

    word=wnl.lemmatize(word)

    if(word in word2weight):#
        return word2weight[word]

    if(word in stop_words):
        return 0.0

    if(word in english_punctuations):#The oov_word is a punctuation
        return 0.0

    if (len(word)<=2):#The oov_word makes no sense
        return 0.0

    if(method=="max_weight"):#Return the max weight of word in the tokenized_sents
        max_weight=0.0
        for w in tokenized_sents:
            if(w in word2weight and word2weight[w] > max_weight):
                max_weight=word2weight[w]
        return max_weight
    return 0.0


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def word_lemmatizer(word):
    tagged_word = nltk.pos_tag([word])[0]
    wnl = WordNetLemmatizer()
    wordnet_pos = get_wordnet_pos(tagged_word[1]) or wordnet.NOUN
    lemma = wnl.lemmatize(tagged_word[0], pos=wordnet_pos)
    return lemma

def splice_embeddings(bert_embeddings, tokens_segment):
    new_bert_embeddings = torch.tensor(bert_embeddings[0][0:len(tokens_segment[0]), : ])
    for i in range(1, len(tokens_segment)):
        embed = torch.tensor(bert_embeddings[i][0: len(tokens_segment[i]), :])
        new_bert_embeddings = torch.cat((new_bert_embeddings, embed), 0)
    return new_bert_embeddings

def get_sent_segmented(tokens):
    min_seq_len = 16
    sents_sectioned = []
    if (len(tokens) < min_seq_len):
        sents_sectioned.append(tokens)
    else:
        pos = 0
        for i, token in enumerate(tokens):
            if token == ".":
                if (i - pos) >= min_seq_len:
                    sents_sectioned.append(tokens[pos: i + 1])
                    pos = i + 1
        if len(tokens[pos:]) > 0:
            sents_sectioned.append(tokens[pos:])
    return sents_sectioned

        
class WordEmbeddings():
    """
    使用bertsentence获取word embedding 和 sentence embedding 
    """
    def __init__(self, model):
        self.sentence_bert = model

    def get_tokenized_words_embeddings(self, sents_tokened):
        bert_embeddings = []
        for i in range(0, len(sents_tokened)):
            length = len(sents_tokened[i]) # 当前句子的长度
            # 对每一个token进行编码 
            be = self.sentence_bert.encode(sents_tokened[i])
            bert_embeddings.append(be)
        return np.array(bert_embeddings)


def SIFRank_plus(text, SIF, method="average", N=15, position_bias = 3.4, openie=False):
    """
    :param text_obj:
    :param sent_embeddings:
    :param candidate_embeddings_list:
    :param sents_weight_list:
    :param method:
    :param N: the top-N number of keyphrases
    :param sent_emb_method: 'elmo', 'glove'
    :param elmo_layers_weight: the weights of different layers of ELMo
    :return:
    """
    text_obj = InputTextObj(text, openie=openie)   # en_model是用于处理词性标注的
    nn_candi_keys = text_obj.nn_keyphrase_candidate
    trples_candi_keys = text_obj.trples_candi_keys
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(text_obj)
    
    position_score = get_position_score(text_obj.nn_keyphrase_candidate, position_bias)
    if len(position_score) > 0:
        average_score = sum(position_score.values()) / (float)(len(position_score))   #Little change here
    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(sent_embeddings, emb)
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')
    for np, dist in dist_final.items():
        if np in position_score:
            dist_final[np] = dist*position_score[np] / average_score#Little change here
    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)

    nn_candi_embed_dict = defaultdict()
    for key, embed in zip(nn_candi_keys, candidate_embeddings_list):
        nn_candi_embed_dict[key[0]] = embed
    return (dist_sorted[0:N], trples_candi_keys, nn_candi_embed_dict)

def get_all_dist(candidate_embeddings_list, text_obj, dist_list):
    '''
    :param candidate_embeddings_list:
    :param text_obj:
    :param dist_list:
    :return: dist_all
    '''
    dist_all={}
    for i, emb in enumerate(candidate_embeddings_list):
        phrase = text_obj.nn_keyphrase_candidate[i][0]
        phrase = phrase.lower()
        phrase = wnl.lemmatize(phrase)
        if(phrase in dist_all):
            #store the No. and distance
            dist_all[phrase].append(dist_list[i])
        else:
            dist_all[phrase]=[]
            dist_all[phrase].append(dist_list[i])
    return dist_all

def get_final_dist(dist_all, method="average"):
    '''
    :param dist_all:
    :param method: "average"
    :return:
    '''

    final_dist={}

    if(method=="average"):

        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            if (phrase in stop_words):
                sum_dist = 0.0
            final_dist[phrase] = sum_dist/float(len(dist_list))
        return final_dist

def get_dist_cosine(emb1, emb2):
    sum = 0.0
    assert emb1.shape == emb2.shape
    return cos_sim(emb1, emb2)

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


def softmax(x):
    # x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def get_position_score(keyphrase_candidate_list, position_bias):
    length = len(keyphrase_candidate_list)
    position_score ={}
    for i, kc in enumerate(keyphrase_candidate_list):
        np = kc[0]
        p = kc[1][0]
        np = np.lower()
        np = wnl.lemmatize(np)
        if np in position_score:

            position_score[np] += 0.0
        else:
            position_score[np] = 1/(float(i)+1+position_bias)
    score_list=[]
    for np, score in position_score.items():
        score_list.append(score)
    score_list = softmax(score_list)

    i=0
    for np, score in position_score.items():
        position_score[np] = score_list[i]
        i+=1
    return position_score


class ClusData:
    
    def __init__(self, clus_data, all_candidates_embed=None):
        # all_candidate_embed是一个关键词的embed字典

        self.keyphrases = clus_data
        assert all_candidates_embed is not None
        self.embed = None
        for w in self.keyphrases:
            if w in all_candidates_embed:
                # 确定每一个embedding的维度
                if self.embed is None:
                    self.embed = all_candidates_embed[w]
                else:
                    self.embed = torch.cat((self.embed, all_candidates_embed[w]), dim=0)
            else:
                self.keyphrases.remove(w)
    
        self.embed = np.array(self.embed)
        
        self.similarity = np.round(cosine_similarity(self.embed), decimals = 5) 
        self.id2kp = {idx:phrase for idx,phrase in enumerate(self.keyphrases)}
        self.kp2id = {j: i for i, j in self.id2kp.items()}



class Clusterer:

    def __init__(self, data, n_cluster=None, distance_threshold = None, method = 'sp-kmeans',affinity = 'euclidean',linkage="average"):
        self.data = data
        self.method = method
        self.n_cluster = n_cluster
        self.distance_threshold = distance_threshold

        # k-means聚类
        if method == 'sp-kmeans':
            self.clus = KMeans(self.n_cluster, random_state = 0)
        
        # 层次聚类
        elif method == 'agglo':
            self.clus=AgglomerativeClustering(n_clusters=n_cluster, distance_threshold=self.distance_threshold, affinity=affinity, linkage=linkage)
        # cluster id -> members
        self.membership = None  # a list contain the membership of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers
        self.class2word = {}
        self.inertia_scores = None
        self.sil_score = None
        self.db_score = None
        
    def fit(self):
        self.clus.fit(self.data.embed)
        labels = self.clus.fit_predict(self.data.embed)
        for idx, label in enumerate(labels):
            if label not in self.class2word:
                self.class2word[label] = []
            self.class2word[label].append(self.data.id2kp[idx])  # 将标签和关键词对应起来
        self.membership = labels
        if self.n_cluster == None:
            self.n_cluster = max(self.class2word.keys())
        if self.method == 'ap':
            self.center_ids = self.gen_center_idx()
        elif self.method == 'sp-kmeans':
            self.center_ids = self.gen_center_idx()
            self.inertia_scores = self.clus.inertia_
            print('Clustering concentration score:', self.inertia_scores)

        self.sil_score = silhouette_score(self.data.embed, self.membership, metric = 'cosine')
        self.db_score = davies_bouldin_score(self.data.embed, self.membership)
        print('Clustering silhouette score:', self.sil_score)
        print('Clustering davies bouldin score:', self.db_score)

    # find the idx of each cluster center
    def gen_center_idx(self):
        ret = []
        for cluster_id in range(self.n_cluster):
            center_idx = self.find_center_idx_for_one_cluster(cluster_id)
            ret.append((cluster_id, center_idx))
        return ret


    def find_opt_k_sil(self, n_cluster):
        sil = []
        kmax = n_cluster
    
        for k in range(2, kmax+1):
            kmeans = SphericalKMeans(n_clusters = k, init='k-means++', random_state = 0, n_init=50, n_jobs = -2).fit(self.data.embed)
            labels = kmeans.labels_
            sil.append(silhouette_score(self.data.embed, labels, metric = 'cosine'))
        
        k_opt = sil.index(max(sil)) + 2
        return k_opt
    
    def find_opt_k_db(self, n_cluster):
        db = []
        kmax = n_cluster
    
        for k in range(2, kmax+1):
            kmeans = SphericalKMeans(n_clusters = k, init='k-means++', random_state = 0, n_init=50, n_jobs = -2).fit(self.data.embed)
            labels = kmeans.labels_
            db.append(davies_bouldin_score(self.data.embed, labels))
        
        k_opt = db.index(min(db)) + 2
        return k_opt
       
    def find_center_idx_for_one_cluster(self, cluster_id):
        query_vec = self.clus.cluster_centers_[cluster_id]
        members = self.class2word[cluster_id]   # 找到当前簇的所有的id
        best_similarity, ret = -1, -1
        for member in members:
            member_idx = self.data.kp2id[member]
            member_vec = self.data.embed[member_idx]
            cosine_sim = self.calc_cosine(query_vec, member_vec)[0][0]
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return ret

    def calc_cosine(self, vec_a, vec_b):
        vec_a = vec_a.reshape(1, -1)
        vec_b = vec_b.reshape(1, -1)
        return cosine_similarity(vec_a, vec_b)



def cluster_keys(key2embed):
 
    content_strings = []
    content_embeds = []
    for key, embed in key2embed.items():
        content_strings.append(key)
        content_embeds.append(embed)
    n_clusters = int(np.ceil(len(content_strings)**0.5))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(content_embeds)

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, content_embeds)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    keys = [content_strings[closest[idx]] for idx in ordering]

    return keys


def cluster(tup_keys, nn_candi_embed_dict):
    nn_keyphrase = []
    for key in tup_keys:
        nn_keyphrase.append(key[0])
    kpterms = ClusData(nn_keyphrase, nn_candi_embed_dict)
    spclus = Clusterer(kpterms, n_cluster=5, method = 'sp-kmeans')
    spclus.fit()
    center_names = []
    clus_centers = spclus.center_ids
    for cluster_id, keyword_idx in clus_centers:
        keyword = kpterms.id2kp[keyword_idx]
        center_names.append(keyword)
    return center_names
    # clusters = {}
    # for i in spclus.class2word:
    #     clusters[float(i)] = find_most_similar(center_names[i], kpterms, spclus, n='all')
    # for i in spclus.class2word:
    #     print(i)
    #     print(center_names[i])
    #     print(spclus.class2word[i])
    #     print('=================================================================')

def find_most_similar(kp, data, clus, n='all'):
    
    kpid = data.kp2id[kp]
    num = clus.membership[kpid]
    distance = 1-data.similarity
    idx = data.kp2id[kp]
    if n == 'all':
        topn = [data.id2kp[i] for i in distance[idx].argsort()[::] if data.id2kp[i] in clus.class2word[num]]
    else:
        topn = [data.id2kp[i] for i in distance[idx].argsort()[::] if data.id2kp[i] in clus.class2word[num]][:n]
        
    return topn


# if __name__ == "__main__":
#     file_path = "/home/jazhan/code/document2slides-main/d2s_model/0.json"
#     model_name = 'distilbert-base-nli-mean-tokens'
#     # en_model = StanfordCoreNLP(r'/home/jazhan/stanford-corenlp-4.2.0', quiet=True)
#     BERT = WordEmbeddings(model_name)
#     SIF = SentEmbeddings(BERT, lamda=1.0)

#     prs = Presentation()
#     slide_num = 0
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     title = data['title'].strip()
#     create_pptx.add_titile(prs, text=title)
#     slide_num += 1
#     for section_text in data["text"]:
#         cur_section = section_text["section"]
#         print("cur_section", cur_section)
#         sub_title = cur_section
        
#         n = section_text['n']   # 章节编号
#         content_string = section_text['strings']
        
#         # keyphrase是最终筛选出来TOPN的关键词元组
#         keyphrase, triple_keyphrase, nn_candi_embed_dict = SIFRank_plus(content_string, SIF, N=30)
        
#         center_keys = cluster(keyphrase, nn_candi_embed_dict)
#         # 聚类的中心词
#         for ck in center_keys:
#             print("ck", center_keys)
#         break
#         # pptx_dict = {"名词短语": "|".join(nn_keyphrase), "关系三元组": "||".join(triple_keyphrase)}
#         pptx_dict = {"名词短语": "|".join(nn_keyphrase)}
#         create_pptx.add_slide(prs, slide_num, title_text=cur_section, tf_content=pptx_dict)
    
#     prs.save('test.pptx')
