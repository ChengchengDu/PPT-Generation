from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch



def generate_summary(src_text):
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch, max_length=200, min_length=20, num_beams=6, length_penalty=2.0, no_repeat_ngram_size=3)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    tgt_text = tgt_text[0].replace("We", "This paper").split('<n>')
    tgt_text = " ".join(tgt_text)

    # print(tgt_text)
    return tgt_text


# if __name__ == "__main__":
#     # src_text = ["n mWe present STARC (Structured Annotations for Reading Comprehension), a new annotation framework for assessing reading comprehension with multiple choice questions. Our framework introduces a principled structure for the answer choices and ties them to textual span annotations. The framework is implemented in OneStopQA, a new high-quality dataset for evaluation and analysis of reading comprehension in English. We use this dataset to demonstrate that STARC can be leveraged for a key new application for the development of SAT-like reading comprehension materials: automatic annotation quality probing via span ablation experiments. We further show that it enables in-depth analyses and comparisons between machine and human reading comprehension behavior, including error distributions and guessing ability. Our experiments also reveal that the standard multiple choice dataset in NLP, RACE (Lai et al., 2017) , is limited in its ability to measure reading comprehension. 47% of its questions can be guessed by machines without accessing the passage, and 18% are unanimously judged by humans as not having a unique correct answer. OneStopQA provides an alternative test set for reading comprehension which alleviates these shortcomings and has a substantially higher human ceiling performance. 1"]
#     src_text = ["This paper investigates contextual word representation models from the lens of similarity analysis. Given a collection of trained models, we measure the similarity of their internal representations and attention. Critically, these models come from vastly different architectures. We use existing and novel similarity measures that aim to gauge the level of localization of information in the deep models, and facilitate the investigation of which design factors affect model similarity, without requiring any external linguistic annotation. The analysis reveals that models within the same family are more similar to one another, as may be expected. Surprisingly, different architectures have rather similar representations, but different individual neurons. We also observed differences in information localization in lower and higher layers and found that higher layers are more affected by fine-tuning on downstream tasks. 1 * Equal contribution 1 The code is available at https://github.com/ johnmwu/contextual-corr-analysis."]
#     # src_text = ["Learning from small data sets is difficult in the absence of specific domain knowledge. We present a regularized linear model called STEW, which benefits from a generic and prevalent form of prior knowledge: feature directions. STEW shrinks weights toward each other, converging to an equalweights solution in the limit of infinite regularization. We provide theoretical results on the equalweights solution that explains how STEW can productively trade-off bias and variance. Across a wide range of learning problems, including Tetris, STEW outperformed existing linear models, including ridge regression, the Lasso, and the non-negative Lasso, when feature directions were known. The model proved to be robust to unreliable (or absent) feature directions, outperforming alternative models under diverse conditions. Our results in Tetris were obtained by using a novel approach to learning in sequential decision environments based on multinomial logistic regression."]
#     generate_summary(src_text)