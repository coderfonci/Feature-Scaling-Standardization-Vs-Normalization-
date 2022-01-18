Bottom-up approaches
Of course, we can also code the equations for standardization and 0-1 Min-Max scaling “manually”. However, the scikit-learn methods are still useful if you are working with test and training data sets and want to scale them equally.

E.g.,

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)
Below, we will perform the calculations using “pure” Python code, and an more convenient NumPy solution, which is especially useful if we attempt to transform a whole matrix.

Just to recall the equations that we are using:

Standardization:

z=x−μσ
with mean:

μ=1N∑i=1N(xi)
and standard deviation:

σ=1N∑i=1N(xi−μ)2
Min-Max scaling:

Xnorm=X−XminXmax−Xmin
