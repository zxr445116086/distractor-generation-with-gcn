#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import ujson as json
import copy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import onmt.opts
from eval.eval import eval

def loads_json(loadpath, loadinfo):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print(loadinfo)
        dataset = []
        for line in fh:
            example = json.loads(line)
            dataset.append(example)
        print('load json done')
    return dataset

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def main(opt):
    testset = loads_json(opt.data, 'load test file')
    f = open("data/corpus.json", "r")
    data = f.readlines()
    samples = []
    for i in data:
        samples.append(json.loads(i)["sents"])
    corpus = []
    for i in samples:
        corpus.append(" ".join([" ".join(sent) for sent in i]))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit(corpus)

    translator = build_translator(opt, report_score=False)
    translated = translator.translate(data_path=opt.data, 
                                      batch_size=opt.batch_size,
                                      tfidf=tfidf)

    # find first 3 not similar distractors
    hypothesis = {}
    for translation in translated:
        sample_id = str(translation.ex_raw.id).split("_")
        question_id = sample_id[0] + "_" + sample_id[1]
        pred1 = translation.pred_sents[0]
        pred2, pred3 = None, None
        for pred in translation.pred_sents[1:]:
            if jaccard_similarity(pred1, pred) < 0.5:
                if pred2 is None:
                    pred2 = pred
                else:
                    if pred3 is None:
                        if jaccard_similarity(pred2, pred) < 0.5:
                            pred3 = pred
            if pred2 is not None and pred3 is not None:
                break

        if pred2 is None:
            pred2 = translation.pred_sents[1]
            if pred3 is None:
                pred3 = translation.pred_sents[2]
        else:
            if pred3 is None:
                pred3 = translation.pred_sents[1]

        hypothesis[question_id] = [pred1, pred2, pred3]
    torch.save(hypothesis, "translated.hypothesis.pt")

    reference = {}
    f = open(opt.output, "w")
    for sample in testset:
        line = copy.deepcopy(sample)
        sample_id = sample["id"].split("_")
        question_id = sample_id[0] + "_" + sample_id[1]
        if question_id not in reference.keys():
            reference[question_id] = [sample['distractor']]
        else:
            reference[question_id].append(sample['distractor'])
        line["pred"] = hypothesis[question_id]
        f.write(json.dumps(line) + "\n")
    f.close()

    _ = eval(hypothesis, reference)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
