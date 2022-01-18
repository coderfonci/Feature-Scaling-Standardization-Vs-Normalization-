# Feature-Scaling-Standardization-Vs-Normalization-
What is Feature Scaling? 
•Feature Scaling is a method to scale numeric features in the same scale or range (like:-1 to 1,  0 to 1). 
•This is the last step involved in Data Preprocessing and before ML model training.  
•It is also called as data normalization.  
•We apply Feature Scaling on independent variables.  
•We fit feature scaling with train data and transform on train and test data.  
Why Feature Scaling? 
#•The scale of raw features is different according to its units.  
•Machine Learning algorithms can’t understand features units, understand only numbers.  
•Ex: If hight 140cm and 8.2feet  
•ML Algorithms understand numbers then 140 > 8.2  Which ML Algorithms Required Feature Scaling? Those Algorithms Calculate Distance •K-Nearest Neighbors (KNN)  
•K-Means  •Support Vector Machine (SVM)  
•Principal Component Analysis(PCA)  
•Linear Discriminant Analysis  Gradient Descent Based Algorithms 
•Linear Regression,  
•Logistic Regression  
•Neural Network  Tree Based Algorithms not required Feature scaling 
•Decision Tree, Random Forest, XGBoost  Types of Feature Scaling 
•1) Min Max Scaler  
•2) Standard Scaler  
•3) Max Abs Scaler  
•4) Robust Scaler  
•5) Quantile Transformer Scaler  
•6) Power Transformer Scaler  
•7) Unit Vector Scaler  Standardization vs Normalization Explain in Detail What is Standardization? 
•Standardization rescale the feature such as mean(μ) = 0 and standard deviation (σ) = 1.  
•The result of standardization is Z called as Z-score normalization.  
• If data follow a normal distribution (gaussian distribution). 
• If the original distribution is normal, then the standardized distribution will be normal.  
• If the original distribution is skewed, then the standardized distribution of the variable will also be skewed.   
What is Normalization? 
•Normalization rescale the feature in fixed range between 0 to 1.  
•Normalization also called as Min-Max Scaling.  
•If data doesn’t follow normal distribution (Gaussian distribution).  
Standardization  vs Normalization 
•There is no any thumb rule to use Standardization or Normalization for special ML algo.  
•But mostly Standardization use for clustering analyses, Principal Component Analysis(PCA).  
•Normalization prefers for Image processing because of pixel intensity between 0 to 255, neural network algorithm requires data in scale 0-1, K-Nearest Neighbors.
