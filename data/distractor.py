import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import json
import re
import sys
import copy

path = sys.argv[1]

def get_article_dict(article):
  d = {}
  punc = {",", ".", "?", "!", "(", ")", "\"", "-"}
  for sent in article:
    for word in sent:
      if word not in STOP_WORDS and word not in punc:
        if word in d:
          d[word] += 1
        else:
          d[word] = 1
  return d

nlp = spacy.load('en_core_web_sm')
f = open(path, "r")
data = f.readlines()
samples = []
for i in data:
  samples.append(json.loads(i))
path = path.split("/")[-1]
g = open("data/distractor4/" + path, "w")
ans = {"A":0,"B":1,"C":2,"D":3}

for sample in samples:
  new_sample = copy.deepcopy(sample)
  del new_sample["answers"]
  del new_sample["options"]
  del new_sample["questions"]
  del new_sample["article"]
  text = sample["article"].replace("\n", " ").replace("\"", " \"")
  text = re.sub(r"([,])([a-zA-Z])", r"\1"+" "+ r"\2", text)
  doc = nlp(text)
  sents = []
  flag = False
  for i, sent in enumerate(doc.sents):
    t = sent.text.strip()
    if flag:
      t = "\"" + t
      flag = False
    if sent.text.strip() == "\"":
      flag = True
      continue
    s = nlp(t)
    tokens = [token.text.strip() for token in s]
    if len(tokens) > 0:
      sents.append(tokens)
  d = get_article_dict(sents)
  new_sample["sents"] = sents
  questions = sample["questions"]
  for idx, question in enumerate(questions):
    ques = question.replace("\"", " \"")
    q = []
    if "_" in ques:
      q = ques.split("_")
      if len(q) > 2:
        continue
    if len(q) == 2:
      a = re.search(r"[a-zA-Z0-9]", q[1])
      if a is not None or len(q[0]) < 1:
        continue
      ques = q[0]
    doc = nlp(ques)
    ques = [token.text.strip() for token in doc]
    new_sample["question"] = ques
    q_id = sample["id"].split(".")[0] + "_q" + str(idx)
    ans_opt = ans[sample["answers"][idx]]
    answer = sample["options"][idx][ans_opt].replace("\"", " \"")
    doc = nlp(answer)
    answer = [token.text.strip() for token in doc]
    new_sample["answer"] = answer
    for index in range(4):
      if index != ans_opt:
        distractor = sample["options"][idx][index].replace("\"", " \"")
        d_id = q_id + "_d" + str(index)
        doc = nlp(distractor)
        distractor = [token.text.strip() for token in doc]
        cnt = 0
        for i in distractor:
          cnt += d.get(i, 0)
        if cnt < 4:
          continue
        new_sample["distractor"] = distractor
        new_sample["id"] = d_id
        g.write(json.dumps(new_sample) + "\n")
g.close()
