import argparse
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import pandas as pd


#Cmd line parsing-------------------------------------------------------------
def cmdLineParsing():
  parser = argparse.ArgumentParser()
  parser.add_argument("--f", help="input data file")
  #parser.add_argument("--n", help="nb of input data", type=int)
  #parser.add_argument("--k", help="number of clusters", type=int)
  parser.add_argument("--c", help="display centroid", action="store_true")

  args = parser.parse_args()

  if not os.path.isfile(args.f):
    sys.exit("Error: input data file '" + args.f + "' is unreachable!")
  #if args.n <= 0:
  #  sys.exit("Error: nb of input data must be greater than 0!")
  #if args.k <= 0:
  #  sys.exit("Error: nb of clusters must be greater than 0!")

  return args.f, args.c  #, args.n, args.k)
  

# Main code-------------------------------------------------------------------
# parse the cmd line
fname, c_true = cmdLineParsing()

# read data instances---------------------------------------------------------
df_instances = pd.read_csv(fname, header=None, names=['x', 'y'], delimiter='\t')

# read labels-----------------------------------------------------------------
df_labels = pd.read_csv("Labels_"+fname, header=None, names=['label'])
df = pd.concat([df_instances, df_labels], axis=1)

df_centroids = pd.read_csv("Centroids_"+fname, header=None, names=['centroids_x', 'centroids_y'], sep='\s+')


# plot the points in different colors and markers according to their labels
colors = ["green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "darkred", "darkgreen", "darkorange", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray", 
          "tomato", "lime", "cornflowerblue", "bisque", "magenta", "darkturquoise", "sandybrown", "hotpink", "darkkhaki", "silver",
          "lightcoral", "lightgreen", "dodgerblue", "wheat", "orchid", "lightseagreen", "darksalmon", "palevioletred", "gold", "dimgray"]
markers = ["s", "v", "o", "^", "<", ">", "p", "P", "*", "X", 
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "X", "*", "P", "p", ">", "o", "s", "v", "^", "<",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1"]

size = 10

for label, group in df.groupby('label'):
    plt.scatter(group['x'], group['y'], c=colors[label], marker=markers[label], s=size)

if c_true is True:
  for i, row in df_centroids.iterrows():
    plt.scatter(row['centroids_x'], row['centroids_y'], c=colors[i], marker=markers[i], s=size*10, label=str(i))
  plt.legend()

plt.savefig('Clustering_' + fname+ '.png', dpi = 400)
