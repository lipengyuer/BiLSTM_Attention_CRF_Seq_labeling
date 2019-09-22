
def get_ngrams(text, N=2, step=1):
    ngrams = []
    for i in range(0, len(text)-N, step):
        ngrams.append(text[i: i+N])
    return ngrams
        