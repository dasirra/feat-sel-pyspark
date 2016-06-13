
from pyspark.mllib.stat import Statistics

def dist_corr(v1, v2):

    return Statistics.corr(v1,v2)

def dist_ftest(v, t):
    
    # calculate auxiliary variables
    n_samples = v.count()
    n_groups = t.distinct().count() # number of distinct groups
    overall_mean = vector.mean() # overall mean
    aux_mean = z_vector.aggregateByKey((0,0),
                                  lambda x,y: (x[0]+y,x[1]+1),
                                  lambda x,y: (x[0]+y[0],x[1]+y[1]))
    group_count = aux_mean.map(lambda (label,x): (label,x[1])) 
    group_mean = aux_mean.map(lambda (label,x): (label,x[0]/x[1])) 
    aux_within = z_vector.leftOuterJoin(group_mean)

    # between-group variability
    num = sum([nx[1]*(mx[1]-overall_mean)**2
        for (nx,mx) in zip(group_count.collect(),
            group_mean.collect())])/float(n_groups-1)
    
    # within-group variability
    den = aux_within.map(lambda (_,x): (x[0]-x[1])**2) \
        .reduce(add)/float(n_samples-n_groups)
    
    return num/den
