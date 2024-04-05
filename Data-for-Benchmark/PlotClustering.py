import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


#Cmd line parsing-------------------------------------------------------------
def cmdLineParsing():
  parser = argparse.ArgumentParser()
  parser.add_argument("datafile", help="input data file")

  args = parser.parse_args()

  if not os.path.isfile(args.datafile):
    sys.exit("Error: input data file '" + args.datafile + "' is unreachable!")

  return args.datafile 
  

# Main code-------------------------------------------------------------------
# parse the cmd line
fname = cmdLineParsing()

# read data instances---------------------------------------------------------
fo = open(fname, 'r')            # Input data file
lines_instances = fo.readlines()
fo.close()

# read labels-----------------------------------------------------------------
fo = open("Labels_"+fname, 'r')  # Label file: "Labels_<file name>"
lines_labels = fo.readlines()
fo.close()

# initialize three arrays dedicated to store the coordinates and the cluster label of each data instance
x = []
y = []
label = []

# scan the rows stored in lines, and split the values into three arrays (x, y, label)
for line in lines_instances:
    p = line.split()
    x.append(float(p[0]))
    y.append(float(p[1]))

for line in lines_labels:
    p = line.split()
    label.append(int(p[0]))


# plot the points in different colors and markers according to their labels
colors = ["green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "darkred", "darkgreen", "darkorange", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray", 
          "tomato", "lime", "cornflowerblue", "bisque", "magenta", "darkturquoise", "sandybrown", "hotpink", "darkkhaki", "silver",
          "lightcoral", "lightgreen", "dodgerblue", "wheat", "orchid", "lightseagreen", "darksalmon", "palevioletred", "gold", "dimgray"]
markers = ["s", "v", "o", "^", "<", ">", "p", "P", "*", "X", 
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "X", "*", "P", "p", ">", "o", "s", "v", "^", "<",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1"]

for i in range(len(label)):
    plt.scatter(x[i], y[i], c = colors[label[i]], marker = markers[label[i]], s = 10)

#plt.show()
plt.savefig('Clustering_' + fname+ '.png', dpi = 400)
