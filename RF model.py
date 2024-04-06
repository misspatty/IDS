import pandas,numpy

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

cic = pandas.read_parquet('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/cic-collection.parquet', engine='pyarrow')

#Randomly select 500,000 data from the dataset
dataSamples = cic.sample(n=500000)

#Categorize malicious network into binary 1 and 0 if benign from ClassLabel column
dataSamples['Category'] = dataSamples['ClassLabel'].apply(lambda x: 1 if x == 1 else 0)

Encoded = LabelEncoder()
dataSamples['Label'] = Encoded.fit_transform(dataSamples['Label'])

#Drop column Label, ClassLabel
DropFeat = ['Label', 'ClassLabel']
X = dataSamples.drop(columns=DropFeat)
Y = dataSamples['Label']

# Perform Random Oversampling
OverSample = RandomOverSampler(random_state=42)
X_os, Y_os = OverSample.fit_resample(X, Y)

# Perform Random Undersampling
UnderSample = RandomUnderSampler(random_state=42)
X_combi, Y_combi = UnderSample.fit_resample(X_os, Y_os)

# Initialize MinMaxScaler
Scale = MinMaxScaler()
X_mms = Scale.fit_transform(X_combi)

# Feature Selection/Dimensionality Reduction (using PCA)
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_mms)

X_train, X_test, Y_train,Y_test = train_test_split(X_pca, Y_combi, test_size=0.30, random_state=42)

RFModel= RandomForestClassifier(n_estimators=100, random_state=42)
RFModel.fit(X_train, Y_train)

#Evaluate the Model
Y_pred = RFModel.predict(X_test)

# Generate a classification report
result = classification_report(Y_test, Y_pred)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='micro')  # Use micro averaging
recall = recall_score(Y_test, Y_pred, average='micro')  # Use micro averaging
f1 = f1_score(Y_test, Y_pred, average='micro')  # Use micro averaging
support = numpy.unique(Y_test, return_counts=True)[1]

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Support:", support)
print("Random Forrest Model Classification Report:\n", result)
