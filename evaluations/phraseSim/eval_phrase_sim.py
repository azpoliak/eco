import math
import argparse
import numpy as np
from sets import Set
from sklearn import manifold
#import pylab as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cocoonEmbeds = {}
rightEmbeds = {}
leftEmbeds = {}
baseLineEmbeds = {}
_dim_ = None
_dim_div_ = None
_window_ = None
embeds_base = "PATH to baseline embeddings"
'''
Fast cosine_similarit method from http://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
'''
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def load_baseline_embeds(base_f, dim, window):
    #baseline.mincount~5.dim~100.window~2.embeds
    print "Opening embed file"
    embed_file = open(base_f + "baseline.mincount~5.dim~" + str(dim) + ".window~" + str(window) + ".embeds", "rb")
    assert dim == int(embed_file.readline().split()[1])
    for line in embed_file:
       lineList = line.split()
       if lineList[0] in FULLVOCAB or lineList[0] == '<unk>':
           baseLineEmbeds[lineList[0]] = lineList[1:]
    #assert len(baseLineEmbeds) == len(FULLVOCAB)

def load_embeds(base_f, dim_div, dim):
    base = 'cocoon.mincount~5.dim~' + str(dim)#.window~-1.dim_divide~10.embeds
    win = (dim_div)/2

    for _ in range(-win, win+1):
        if _ == 0:
            continue
        print "Opening " + str(_) + "R"
        r_file = open(base_f + base + ".window~" + str(_) + ".dim_divide~" + str(dim_div) + ".embeds", "rb")
        #skip first line of file which is just metadata (number of words and dimensionality of embeddings#
        assert (dim / dim_div) == int(r_file.readline().split()[1])

        cocoonEmbeds[_] = {}
        for line in r_file.readlines():
            lineList = line.split()
            if lineList[0] in FULLVOCAB or lineList[0] == '<unk>':
                cocoonEmbeds[_][lineList[0]] = lineList[1:]
        r_file.close()

        #assert (cocoonEmbeds[_]) == len(FULLVOCAB)

    load_baseline_embeds(base_f, dim, dim_div / 2)


def make_plot(d2, phrases, filename):
    colors = ['red', 'green', 'blue', 'cyan', 'orange', 'purple', 'red']
    usePhrases = Set()
    x, y, c, l = [],[],[], []
    for i in range(len(d2[0])):
        if (phrases[i][0], phrases[i][0]) in usePhrases or (phrases[i][1], phrases[i][0]) in usePhrases:
            continue
        usePhrases.add((phrases[i][0], phrases[i][1]))
        x.append(d2[0][i,0] * 10)
        y.append(d2[0][i,1] * 10)
        l.append(phrases[i][0])
        c.append(colors[i % len(colors)])
        plt.text(x[-1], y[-1], l[-1])

        x.append(d2[1][i,0] * 10)
        y.append(d2[1][i,1] * 10)
        l.append(phrases[i][1])
        c.append(colors[i % len(colors)])
        plt.text(x[-1], y[-1], l[-1])
        #c.append(soundsColorMap[trainLabels[0][i][0]])

    plt.scatter(x,y,color=c, label=l)
    plt.title(filename)
    #plt.show()
    #plt.axis([-2000,2000,-500,500])
    plt.savefig(filename + '.png')

    plt.axis([-2000,2000,-500,500])
    plt.savefig(filename + 'zoom.png')

    plt.close()

def plot_embds(nonUnk):
    tsne = manifold.TSNE(n_components=2)
    phrases = [(obj.source, obj.target) for obj in nonUnk]

    embds = [obj.source_embed for obj in nonUnk]
    d2Source = tsne.fit_transform(embds)
    embds = [obj.target_embed for obj in nonUnk]
    d2Target = tsne.fit_transform(embds)
    make_plot((d2Source, d2Target), phrases, "cocoon_both")

    embds = [obj.source_baseline_embed for obj in nonUnk]
    d2Source = tsne.fit_transform(embds)
    embds = [obj.target_baseline_embed for obj in nonUnk]
    d2Target = tsne.fit_transform(embds)
    make_plot((d2Source, d2Target), phrases, "baseline")

def embed_baseline(phrase, dim=None):
        tmpkey = next(baseLineEmbeds.iterkeys())
        dim = dim if dim is not None else len(baseLineEmbeds[tmpkey])
        phraseList = phrase.split()
        wordVec = np.zeros(dim)
        unk = False
        for _ in range(1, len(phraseList) + 1):
            currWord = phraseList[_-1] if phraseList[_-1] in baseLineEmbeds else "<unk>"
            if currWord == "<unk>":
                unk = True
            wordVec += np.fromstring(' '.join(baseLineEmbeds[currWord]), dtype=float, sep= ' ')
        return wordVec / len(phraseList)

def embed_cocoon2(phrase):
	dim_by_dim_div=len(cocoonEmbeds[-1]["<unk>"])

def embed_cocoon(phrase, dim=_dim_, window=_window_, dim_div=_dim_div_):
        phraseList = phrase.split()
        unkCount = 0
        left = np.zeros(dim / dim_div)
        right = np.zeros(dim / dim_div)
        unk = False
        for _ in range(1, len(phraseList) + 1):
            subScript = _ if _ < window else window
            rightSubScript = len(phrase) - _ + 1 if len(phrase) - _ + 1 < window else window
            currWord = phraseList[_-1] if phraseList[_-1] in cocoonEmbeds[-2] else "<unk>"
            if currWord == "<unk>":
                unk = True
                unkCount += 1
            left += np.fromstring(' '.join(cocoonEmbeds[-subScript][currWord]), dtype=float, sep= ' ')
            right += np.fromstring(' '.join(cocoonEmbeds[rightSubScript][currWord]), dtype=float, sep= ' ')

        #average left and right and append
        left = left / len(phraseList)
        right = right / len(phraseList)

        return np.append(right, left)#left, right)#, unkCount


def evaluate(dim, window, ppdb_file, dim_div):
    f = open(ppdb_file, "rb")
    baseline_c_lex = 0
    cocoon_c_lex = 0
    tot_lex = 0

    baseline_c_phrasal = 0
    cocoon_c_phrasal = 0
    tot_phrasal = 0
    for line in f:
        baseline_c = 0
        cocoon_c = 0
        tot = 0
        lineList = line.split('\t')
        source = lineList[0]
        target1 = lineList[1]
        target2 = lineList[2]
        score = int(lineList[3].strip())

        #Compute baseline
        baselineSource = embed_baseline(source, dim)
        baselineTarget1 = embed_baseline(target1, dim)
        baselineTarget2 = embed_baseline(target2, dim)

        score1 = cosine_similarity(baselineSource, baselineTarget1)
        score2 = cosine_similarity(baselineSource, baselineTarget2)

        cocoonSource = embed_cocoon(source, dim, window, dim_div)
        cocoonTarget1 = embed_cocoon(target1, dim, window, dim_div)
        cocoonTarget2 = embed_cocoon(target2, dim, window, dim_div)
        c_score1 = cosine_similarity(cocoonSource, cocoonTarget1)
        c_score2 = cosine_similarity(cocoonSource, cocoonTarget2)

        if score == 0:
            if math.fabs(score1 - score2) < 0.0001:
                baseline_c += 1
            if math.fabs(c_score1 - c_score2) < 0.0001:
                cocoon_c += 1
        elif score == 1:
            if score1 > score2:
                baseline_c += 1
            if c_score1 > c_score2:
                cocoon_c += 1
        elif score == 2:
            if score2 > score1:
                baseline_c += 1
            if c_score2 > c_score1:
                cocoon_c += 1
        if "phrasal" in lineList[4]:
            baseline_c_phrasal += baseline_c
            cocoon_c_phrasal += cocoon_c
            tot_lex +=1
        else:
            baseline_c_lex += baseline_c
            cocoon_c_lex += cocoon_c
            tot_phrasal +=1

    print "Baseline Accuracy: Phrasal: " + str(float(baseline_c_phrasal) / tot_phrasal)
    print "Cocoon Accuracy: Phrasal: " + str(float(cocoon_c_phrasal) / tot_phrasal)

    print "Baseline Accuracy: Lexical: " + str(float(baseline_c_lex) / tot_lex)
    print "Cocoon Accuracy: Lexical: " + str(float(cocoon_c_lex) / tot_lex)

    print "Baseline Accuracy: " + str(float(baseline_c_lex + baseline_c_phrasal) / (tot_phrasal + tot_lex))
    print "Cocoon Accuracy: " + str(float(cocoon_c_lex + cocoon_c_phrasal) / (tot_phrasal + tot_lex))

def load_vocab(ppdb_file):
    vocab = Set()
    f = open(ppdb_file, 'rb')
    for line in f:
        for phrase in line.split('\t')[0:3]:
            for word in phrase.split():
                vocab.add(word.strip())
    return vocab

def setup(dim, dim_div, vocab_set):
    ''' Function to setup evertyhing:
    @params dim: dimensionality of the file
    @params dim_div: the size of the cocoon embeddings
    @params vocab_set: a set of words, i.e. vocab
    '''
    global cocoonEmbeds
    cocoonEmbeds = {}
    global rightEmbeds
    rightEmbeds = {}
    global leftEmbeds
    leftEmbeds = {}
    global baseLineEmbeds
    baseLineEmbeds = {}
    global FULLVOCAB
    FULLVOCAB = vocab_set

    global _dim_
    _dim_ = int(dim)
    global _dim_div_
    _dim_div_ = int(dim_div)
    global _window_
    _window_ = _dim_div_ / 2
    load_embeds(embeds_base, int(dim_div), int(dim))

def project(phrase, baseline=True):
    ''' Function to embed a phrase
    @param phrase: the phrase to params. Should be passed as a string
    @param baseline: True if you want baseline phrase embedding, False if you want cocoon phrase embedding
    '''
    if baseline:
       return embed_baseline(phrase)
    return embed_cocoon(phrase)

def main(embd_file_base, window, ppdb_file):
    global FULLVOCAB
    FULLVOCAB = load_vocab(ppdb_file)
    for dim in [100, 500, 700]:
        for dim_div in [4]: #, 10]:
            print str(dim) + " " + str(dim_div)
            load_embeds(embd_file_base, dim_div, dim)
            evaluate(dim, dim_div / 2, ppdb_file, dim_div)
    #Extract ppdb objects
    #plot_embds(nonUnk)

    #create embeddings based on embd_file_base + window for the ppdb object
    #add embeddings to ppdb objects
    #compute cosine distance between phrases in a ppdb object
    #create visualization based on phrases


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-embeddings', help='Base name of embeddings file', dest='embd_base', default=embeds_base)
    parser.add_argument('-window', help='Max window length. Used to extract the embeddings files', dest='win', default=2, type=int)
    parser.add_argument('-ppdb', help='File containing eval phrases from ppdb', dest='ppdbFile', default='data/clean.gold')
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=8, type=int)
    args = parser.parse_args()

    main(args.embd_base, args.win, args.ppdbFile)
