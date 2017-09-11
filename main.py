"""
    Natural Language Processing Lab
    Hidden Markov Model (HMM) and Part of Speech (POS) Tagging
    Due: 16.04.2017
"""
import glob
import os
import re


def main():
    # Observations (Emissions) - wordpairs {(tag,word):count}
    # Transitions - tagpairs {(tag1,tag2):count}
    wordpairs, tagpairs, tags = get_words("./brown/")
    # print_txt(wordpairs, "wordpairs.txt")
    # print_txt(tagpairs, "tagpairs.txt")
    # print_txt(tags, "tags.txt")

    print("------ TASK 3 ------")
    helper(wordpairs, tagpairs, tags, "task3", "./input/input_tokens.txt", "output_tokens.txt")
    print("------ TASK 4 ------")
    helper(wordpairs, tagpairs, tags, "task4", "./input/test_set.txt", "output_set.txt")


def read_txts(path):
    corpus = ""
    os.chdir(path)
    for file in glob.glob("*"):
        f = open(file, "r")
        corpus += str(f.read()) + "\n"
    os.chdir("..")
    return corpus


def print_txt(item, path):
    f = open(path, "w")
    for key, value in item.items():
        f.write(str(key) + " : " + str(value) + "\n")
    f.close()


def get_words(path):
    corpus = read_txts(path)
    wordpairs = {}              # Observations
    tagpairs = {}               # Transitions
    tags = {}                   # Independent tags

    pattern = r"\t|\n"          # Split into lines and send to the model builder
    corpus2 = re.split(pattern, corpus)
    for line in corpus2:
        if line.isspace() or not line:
            continue
        else:
            line2 = line.lower() + " /"   # Indicate the end of line
            wordpairs, tagpairs, tags = build_model(line2, wordpairs, tagpairs, tags)
    return wordpairs, tagpairs, tags


def build_model(line, wordpairs, tagpairs, tags):
    tokens = re.split("\s+", line)

    for i in range(0, len(tokens)-1):
        pair1 = tokens[i].rsplit("/", 1)
        if tokens[i+1]:
            pair2 = tokens[i + 1].rsplit("/", 1)

        if pair1[0] and pair1[1]:
            wordpair = (str(pair1[0]), str(pair1[1]))   # Observation (Emission)

            if wordpair in wordpairs:
                wordpairs[wordpair] += 1
            else:
                wordpairs[wordpair] = 1

            if pair1[1] in tags:
                tags[pair1[1]] += 1
            else:
                tags[pair1[1]] = 1

        if pair1[1] and pair2[1]:       # Transition
            tagpair = (pair1[1], pair2[1])
            if tagpair in tagpairs:
                tagpairs[tagpair] += 1
            else:
                tagpairs[tagpair] = 1

    return wordpairs, tagpairs, tags


def helper(wordpairs, tagpairs, tags, task, input, output):
    f = open(input, "r")
    text = (f.read()).lower()
    f.close()
    lines = re.split("\n", text)

    f = open(output, "w")
    if task == "task3":
        for line in lines:
            if line.isspace() or not line:
                continue
            else:
                tokens = re.split("\s+", line)
                result = match_tags(tokens, wordpairs, tags)
                for i in result:
                    f.write(i + " ")
                f.write("\n")
    if task == "task4":
        for line in lines:
            if line.isspace() or not line:
                continue
            else:
                newline = line + " /"
                tokens = re.split("\s+", newline)
                result = viterbi(tokens, wordpairs, tagpairs, tags)
                for i in result:
                    f.write(i + " ")
                f.write("\n")
    f.close()


def match_tags(tokens, wordpairs, tags):
    list = []
    for t in tokens:
        dic = {}
        for key, value in wordpairs.items():        # key = (word,tag)
            if key[0] == t:
                # P(w|t) = C(t,w) / C(t)            # Observation probability
                dic[key[1]] = value / tags.get(key[1])
        if dic:
            best = max(dic, key=dic.get)
            print(str(t) + "/" + str(best))
            list.append(str(t) + "/" + str(best))
    return list


def viterbi(tokens, wordpairs, tagpairs, tags):
    # It stores every tag probability for each observation {(word,tag):probability}
    observations = []

    # Observations
    for index, token in enumerate(tokens):
        if index < len(tokens) - 1:
            dic = {}
            for key, value in wordpairs.items():
                if key[0] == token:
                    # P(w|t) = C(t,w) / C(t)
                    pair = (token, key[1])
                    dic[pair] = value / tags.get(key[1])
            if dic:
                observations.append(dic)

    x = observations.pop()
    last = max(x, key=x.get)
    pos_tags = [last[0] + "/" + last[1]]    # Best tags w.r.t Viterbi algorithm

    # Transitions
    while observations:
        element = observations.pop()
        dic = {}
        for keyb, valueb in element.items():
            for key, value in tagpairs.items():
                if key[1] == keyb[1]:
                    # P(t2 | t1) = C(t1, t2) / C(t1)        # Emission probability
                    dic[keyb] = (value / tags.get(key[0])) * valueb
        best = max(dic, key=dic.get)
        pos_tags.append(best[0] + "/" + best[1])

    for t in reversed(pos_tags):
        print(t)
    return reversed(pos_tags)

if __name__ == "__main__":
    main()
