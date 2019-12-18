import sys
import json

txt = sys.argv[1]

samples = {}
with open(txt, "r") as f:
    data = f.readlines()
    for idx, i in enumerate(data):
        print("-------article {} -------".format(idx))
        line = json.loads(i)
        print("question:")
        print(" ".join(line["question"]))
        print("distractor:")
        print(" ".join(line["distractor"]))
        print("prediction:")
        for pred in line.get("pred", []):
            print(" ".join(pred))
        if idx == 99:
            exit()
        print("--------------------------------------------------")
