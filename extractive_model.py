from functools import lru_cache
import json
from pickle import NEWFALSE
import re
from nltk import data
import nltk
from nltk.sem.evaluate import Model
from nltk.tokenize import sent_tokenize
import numpy as np
from numpy.lib.npyio import save
from numpy.lib.utils import _set_function_name
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import sys
sys.path.append("/home/jazhan/code/document2slides-main/d2s_model/")
sys.path.append("/home/jazhan/code/PacSum-master/code") 
from infer import extract_summary_pac
from extract_keys import WordEmbeddings, SentEmbeddings, SIFRank_plus
from create_pptx import make_slide
from abstract_generate import generate_summary
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def filter_sent(sent):
    sent = re.sub(r'[(].*[)]','',sent)
    return sent

def cluster_summary(content_string, model, n_clusters=4):
    sentences = sent_tokenize(content_string, language='english')
    # You would need to download pre-trained models first
    content_embedding = model.encode(sentences)
    # n_clusters = int(np.ceil(len(content_embedding)**0.5))
    if n_clusters > len(sentences):
        n_clusters = len(sentences)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(content_embedding)

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, content_embedding)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = [sentences[closest[idx]] for idx in ordering]

    return summary


if __name__ == "__main__":
    sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    BERT = WordEmbeddings(model=sentence_model)
    SIF = SentEmbeddings(BERT, lamda=1.0)
    # 965 
    with open('/home/jazhan/code/document2slides-main/input/ACL_paper_json/965.json', 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    extractive_summary_dict = {
        "title": "",
        "abstract": [],
        "extractive_summary":[],
    }
    slide_dict = {
        "paper_title": "",
        "paper_abstract": "",
        "slides":[]
    }
    slide_dict["paper_title"] = data["title"]
    abstract = generate_summary(data["abstract"])
    slide_dict["paper_abstract"] = abstract
    num2sec = {}
    section_names = []
    for section_text in data["text"]:
        slide = []
        section_name = section_text["section"]
        print("section_name")
        print(section_name)
        section_names.append(section_name)

        if section_name.lower() == "related work":
            continue
        n = section_text['n']   # 章节编号
        if len(n.split('.')) == 2:
            n = str(float(n))
        num_sec = n.split('.')[0]
    
        if num_sec not in num2sec:
            num2sec[num_sec] = section_name
        
        # 获取当前章节幻灯片的大标题
        if len(n.split('.')) > 2 or (len(n.split('.')) == 2 and n.split('.')[1] != '0'):
            ps_titl = num2sec[num_sec] + "--" + section_name
        else:     
            ps_titl = num2sec[num_sec] 
        
        content_string = section_text['strings']
        if len(nltk.sent_tokenize(content_string)) < 4:
            continue

        # 先进行一个内容的聚类 获取主要句子
        # summary = cluster_summary(content_string, model=sentence_model)
        # pac模型的抽取
        summary = extract_summary_pac(content_string)
        for sum in summary:
            if len(sum.split()) < 12:
                summary.remove(sum)
                
        # 从聚类出的摘要中抽取关键词作为子标题 可以使用3个关键字作为一个子标题 自定义为四句话
        for i in range(len(summary)):
            cur_sum = summary[i]
            sent_kp_tu, sent_trkp, sent_nn_candi_embed = SIFRank_plus(cur_sum, SIF, N=2, openie=False)
            sent_kp = [tp[0] for tp in sent_kp_tu]
            if len(slide) >= 2:
                slide_dict["slides"].append(slide)
                slide = []
            if len(sent_kp) > 0:
                part_slide = {"ps_title": ps_titl, "sent_kp": sent_kp, "cur_sum": cur_sum}
                slide.append(part_slide)
            
    save_path = "/home/jazhan/code/document2slides-main/d2s_model/demo_pptx/"
    make_slide(slide_dict, save_path=save_path)


