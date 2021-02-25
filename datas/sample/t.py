import pickle
import codecs

with open("test.txt", "rb") as f:
    with codecs.open("test_.txt", "w", encoding="utf-8") as w:
        d = pickle.load(f)
        for tr, tx in zip(d[0], d[1]):
            w.write(" ".join([str(x) for x in tr]) + " " + str(tx) + "\n")
