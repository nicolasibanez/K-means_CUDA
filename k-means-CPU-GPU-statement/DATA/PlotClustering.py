import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

#Cmd line parsing-------------------------------------------------------------
def cmdLineParsing():
  parser = argparse.ArgumentParser()
  parser.add_argument("datafile", help="input data file")
  #parser.add_argument("--n", help="nb of input data", type=int)
  #parser.add_argument("--k", help="number of clusters", type=int)

  args = parser.parse_args()

  if not os.path.isfile(args.datafile):
    sys.exit("Error: input data file '" + args.datafile + "' is unreachable!")
  #if args.n <= 0:
  #  sys.exit("Error: nb of input data must be greater than 0!")
  #if args.k <= 0:
  #  sys.exit("Error: nb of clusters must be greater than 0!")

  return args.datafile  #, args.n, args.k)
  

# Main code-------------------------------------------------------------------
# parse the cmd line
fname = cmdLineParsing()

# read data instances---------------------------------------------------------
# fo = open(fname, 'r')
# lines_instances = fo.readlines()
# fo.close()

# pandas version (tab separated values)
data_df = pd.read_csv(fname, sep='\t', header=None)



# read labels-----------------------------------------------------------------
# fo = open("Labels_"+fname, 'r')
# lines_labels = fo.readlines()
# fo.close()

label_df = pd.read_csv("Labels_"+fname, sep='\t', header=None)

# initialize three arrays dedicated to store the coordinates and the cluster label of each data instance
x = []
y = []
label = []

# scan the rows stored in lines, and split the values into three arrays (x, y, label)
# for line in lines_instances:
#     p = line.split()
#     x.append(float(p[0]))
#     y.append(float(p[1]))

# for line in lines_labels:
#     p = line.split()
#     label.append(int(p[0]))


x = data_df[0].values
y = data_df[1].values
label = label_df[0].values


# plot the points in different colors and markers according to their labels
colors = ["green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "darkred", "darkgreen", "darkorange", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray", 
          "tomato", "lime", "cornflowerblue", "bisque", "magenta", "darkturquoise", "sandybrown", "hotpink", "darkkhaki", "silver",
          "lightcoral", "lightgreen", "dodgerblue", "wheat", "orchid", "lightseagreen", "darksalmon", "palevioletred", "gold", "dimgray"]
markers = ["s", "v", "o", "^", "<", ">", "p", "P", "*", "X", 
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "X", "*", "P", "p", ">", "o", "s", "v", "^", "<",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1"]

print("Plotting ...")
# for i in range(len(label)):
#     plt.scatter(x[i], y[i], c = colors[label[i]], marker = markers[label[i]], s = 10)

# hue :
sns.scatterplot(x=x, y=y, hue=label, palette="deep")
# no legend
# plt.legend([],[], frameon=False)

#plt.show()
plt.savefig('Clustering_' + fname+ '.png', dpi = 400)
