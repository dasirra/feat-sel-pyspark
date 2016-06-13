from operator import add

from pyspark.ml.feature import RFormula

from mathutils import *

class SelectKBest():

	def __init__(self, k=3, method="corr"):

		self.k_ = k

		if method == 'corr':
			self.sfunc_ = dist_corr
		elif method == 'fscore':
			self.sfunc_ = dist_ftest
		else:
			raise ValueError('Invalid method: only corr or fscore')

		self._fitted = False

	def fit(self, df, featureCols, targetCol):

		# build features assemble
		formula = RFormula(
		    formula = '{target} ~ {predictors}'.format(target=targetCol,
		    	predictors='+'.join(featureCols)),
		    featuresCol = 'features',
		    labelCol = 'target'
		)
		assembled_df = formula.fit(df).transform(df)

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

		self._fitted = True

	def transform(self, df):

		if not self._fitted:
			raise RuntimeError('First call fit() method')

		cols = df.columns
		idx = sorted(range(len(self.scores_)),key=self.scores_.__getitem__)

		return df.select(*[cols[idd] for idd in idx])