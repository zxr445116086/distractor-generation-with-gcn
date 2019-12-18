import sys
import json
import os

d = sys.argv[1]
dh = d + "/high"
dm = d + "/middle"
fh = os.listdir(dh)
fm = os.listdir(dm)
g = open(d + ".json", "w")
for i in fh:
  path = os.path.join(dh, i)
  print(path)
  f = open(path, "r")
  data = f.readlines()[0].strip()
  f.close()
  g.write(data + "\n")

for i in fm:
  path = os.path.join(dm, i)
  print(path)
  f = open(path, "r")
  data = f.readlines()[0].strip()
  f.close()
  g.write(data + "\n")
g.close()
