# Author: Daniel Sierra
# GitHub: https://github.com/dasirra/feat-sel-pyspark

from operator import add

from pyspark.ml.feature import VectorAssembler

from mathutils import *

class SelectKBest():
    """Select K Best for feature selection

    Parameters
    ----------
    k : int, (default 3).
        number of features to keep

    method : str, "corr" of "fscore"

    Attributes
    ----------
    scores_ : list, [1, n_samples]
        scores of all features in dataset

    """


    def __init__(self, k=3, method="corr"):

        self.k_ = k

        if method == 'corr':
            self.sfunc_ = dist_corr
        elif method == 'fscore':
            self.sfunc_ = dist_ftest
        else:
            raise ValueError('Invalid method: only corr or fscore')

        self._fitted = False

    def transform(self, df, featureCols, targetCol):
        """Keep the K most important features of the Spark DataFrame

        Parameters
        ----------
        df : Spark DataFrame
        featureCols: array, names of feature columns
            to consider in the feature selectio algorithm
        targetCol: str, name of target column, i.e, column to which
            compare each feature.

        Returns
        -------
        transformed_df : New Spark DataFrame with only the most important
            feature columns.

        """

        # build features assemble
        assembler = VectorAssembler(
            inputCols = featureCols,
            outputCol = 'features')
        assembled_df = assembler.transform(df)

        # rename target column
        assembled_df = assembled_df.withColumnRenamed(targetCol,'target')

        # extract features and target
        feats = assembled_df.select('features').rdd
        feats = feats.map(lambda x: x['features'])
        target = assembled_df.select('target').rdd
        target = target.map(lambda x: x['target'])

        # compute per-column metric
        scores = []
        for i,feat in enumerate(featureCols):
            vector = feats.map(lambda x: x[i])
            scores.append(self.sfunc_(vector,target))
        self.scores_ = scores
        
        # sort scores
        idx = sorted(range(len(self.scores_)),reverse=True,key=self.scores_.__getitem__)
        
        # return dataframe with k-best columns 
        return df.select(*[featureCols[idd] for idd in idx[:self.k_]])
