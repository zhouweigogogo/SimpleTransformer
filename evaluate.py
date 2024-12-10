import math
from collections import Counter
 
 
def cal_precision(reference, candidate, n):
    candidate_ngrams = [tuple(candidate[i:i + n])
                        for i in range(len(candidate) - n + 1)]
    reference_ngrams = [tuple(reference[i:i + n])
                        for i in range(len(reference) - n + 1)]
 
    candidate_ngram_counts = Counter(candidate_ngrams)
    reference_ngram_counts = Counter(reference_ngrams)
 
    # Count the number of n-grams that appear in both candidate and reference
    overlap_ngrams = sum(
        min(candidate_ngram_counts[ngram], reference_ngram_counts[ngram])
        for ngram in candidate_ngram_counts
    )
 
    return overlap_ngrams / len(candidate_ngrams)
 
 
def cal_bleu(reference, candidate, max_n=4):
    if len(candidate) == 0:
        return 0.0
 
    brevity_penalty = 1 if len(candidate) > len(reference) else math.exp(1 - (len(reference) / len(candidate)))
 
    term = math.exp(sum(1 / n * math.log(cal_precision(reference, candidate, n)) for n in range(1, max_n + 1)))
    
    bleu = brevity_penalty * term
 
    return bleu
 
 
if __name__ == "__main__":
    # 定义参考翻译和预测翻译
    reference = ['this', 'is', 'a', 'test']
    candidate = ['this', 'is', 'a', 'test', 'too']
 
    # 计算BLEU分数
    bleu_score = cal_bleu(reference, candidate, max_n=4)
 
    print(f'BLEU Score: {bleu_score}')