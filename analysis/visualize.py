# Some code for visualization of the raw data.

import pandas as pd
from sklearn import ensemble
import pylab
import math

def plot_scatter_all(df_train, columns):
    """
    Makes scatter plots of all combinations of quantative features.
    """
    # Get a feature for the x-axis
    for i in xrange(1,11):
        count = 0
        # Make subplots for all other features.
        for j in xrange(1,11):
             if i != j:
                 x = df_train[columns[i]]
                 y = df_train[columns[j]]
                 z = df_train['Cover_Type']

                 # Make a subplot on position of count.
                 count += 1
                 pylab.subplot(3,3,count)
                 pylab.scatter(x,y, c=z)
                 pylab.xlabel(columns[i])
                 pylab.ylabel(columns[j])

        # Show plot.
        pylab.suptitle("Scatter plots of " + columns[i])
        pylab.show()

def plot_hist(df_train, columns):
    """
    Makes histograms of all quantative features.
    """
    for i in xrange(1,11):
        type_1 = []
        type_2 = []
        type_3 = []
        type_4 = []
        type_5 = []
        type_6 = []
        type_7 = []

        # Make different variables for all the different cover types.
        for j in xrange(len(df_train['Cover_Type'])):
            cover_type = df_train['Cover_Type'][j]
            instance = df_train[columns[i]][j]
            if cover_type == 1:
                type_1.append(instance)
            elif cover_type == 2:
                 type_2.append(instance) 
            elif cover_type == 3:
                 type_3.append(instance)
            elif cover_type == 4:
                 type_4.append(instance) 
            elif cover_type == 5:
                 type_5.append(instance) 
            elif cover_type == 6:
                 type_6.append(instance) 
            elif cover_type == 7:
                 type_7.append(instance)

        names = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
        colors = ["navy", "lime", "saddlebrown", "royalblue", "darkred", "darkorange", "forestgreen"]

        # Plot the histogram.        
        pylab.hist([type_1, type_2, type_3, type_4, type_5, type_6, type_7], 50, label=names, stacked=True, color=colors)
        pylab.xlabel(columns[i])
        pylab.ylabel("Frequency")
        pylab.legend()
        pylab.show()

def entropy1(df_train, columns, n_bins):
    """
    Calculates entropy of quantative features.
    n_bins: number of bins
    """
    for i in xrange(1,11):
        # Get feature and zip it with cover type.
        feature =  df_train[columns[i]]
        cover_type = df_train['Cover_Type']
        together = zip(feature,cover_type)
        sorted_together = sorted(together)

        # Split in even bins.
        bin = 15120/float(n_bins)
        splitted_together = split(sorted_together, int(bin))
        
        # For each bin, get the entropy.
        total_entropy = []
        for j in xrange(0,n_bins):

            # Get cover_types in bin and calculate probability.
            cover_type_bin = [x[1] for x in splitted_together[j]]
            probs = []
            for value in xrange(1,8):
                prob = cover_type_bin.count(value)/bin
                probs.append(prob)

            # Calculate entropy.
            entropy_bin = []
            for p in probs:
                if p == 0.0:
                    ent = 0
                else:
                    ent = -p*math.log(p,2)
                entropy_bin.append(ent)

            # Correct for number of bins.
            total_entropy.append((1/float(n_bins))*sum(entropy_bin))

        print sum(total_entropy), columns[i]
        
def entropy2(df_train, columns):
    """
    Calculates entropy of the two categorical features.
    """
    for i in xrange(11,13):
        # Get feature and zip it with cover type.
        feature = df_train[columns[i]]
        cover_type = df_train['Cover_Type']
        together = zip(feature,cover_type)
        sorted_together = sorted(together)

        # Split based on possible values.
        bins = []
        for val in xrange(1,(max(feature)+1)):
            bin = [x for x in together if x[0] == val]
            bins.append(bin)        

        # For each bin, get the entropy.
        total_entropy = []
        for j in xrange(0,len(bins)):

            # Get cover_types in bin and calculate probability.
            cover_type_bin = [x[1] for x in bins[j]]

            # Set 1.0 in case of empty bin.
            if len(cover_type_bin) == 0:
                length = 1.0
            else:
                length = float(len(cover_type_bin))

            probs = []
            for value in xrange(1,8):
                prob = cover_type_bin.count(value)/length
                probs.append(prob)

            # Calculate entropy.
            entropy_bin = []
            for p in probs:
                if p == 0.0:
                    ent = 0
                else:
                    ent = -p*math.log(p,2)
                entropy_bin.append(ent)

            # Correct for number of bins.
            total_entropy.append((1/float(len(bins)))*sum(entropy_bin))

        print sum(total_entropy), columns[i]
    
    
def split(l,n):
    """
    Splits list in even bins.
    """        
    return [l[i:i+n] for i in xrange(0, 15120, n)]
             

if __name__ == "__main__":
    loc_train = "../data/train_normalised.csv"

    # Read data and get columns.
    df_train = pd.read_csv(loc_train)
    columns = df_train.columns
    
    #plot_scatter_all(df_train, columns)
    #plot_hist(df_train, columns)
    entropy1(df_train, columns, 14)
    entropy2(df_train, columns)

