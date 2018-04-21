from sklearn.externals import joblib
from hmmlearn.hmm import GaussianHMM

model = joblib.load( "model.pkl")

print(model.score())
