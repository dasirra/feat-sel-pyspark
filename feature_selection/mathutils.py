# Author: Daniel Sierra
# GitHub: https://github.com/dasirra/feat-sel-pyspark

from operator import add

from pyspark.mllib.stat import Statistics

def dist_corr(v1, v2):
    """
    Function to compute correlation between two Spark RDDs
    """

    return Statistics.corr(v1,v2)

def dist_ftest(v, t):
    """
    Function to compute F-Score between two Spark RDDs
    """
    
    # calculate auxiliary variables
    n_samples = v.count()
    n_groups = t.distinct().count() # number of distinct groups
    overall_mean = v.mean() # overall mean
    zv = t.zip(v)
    aux_mean = zv.aggregateByKey((0,0),
                                  lambda x,y: (x[0]+y,x[1]+1),
                                  lambda x,y: (x[0]+y[0],x[1]+y[1]))
    group_count = aux_mean.map(lambda (label,x): (label,x[1])) 
    group_mean = aux_mean.map(lambda (label,x): (label,x[0]/x[1])) 
    aux_within = zv.leftOuterJoin(group_mean)

    # between-group variability
    num = sum([nx[1]*(mx[1]-overall_mean)**2
        for (nx,mx) in zip(group_count.collect(),
            group_mean.collect())])/float(n_groups-1)
    
    # within-group variability
    den = aux_within.map(lambda (_,x): (x[0]-x[1])**2) \
        .reduce(add)/float(n_samples-n_groups)
    
    return num/den
