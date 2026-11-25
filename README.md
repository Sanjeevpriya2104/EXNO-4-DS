# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
from google.colab import files
uploaded = files.upload()
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/bmi.csv')
df.head()
```
<img width="828" height="312" alt="image" src="https://github.com/user-attachments/assets/c5fb7280-b168-4d3e-b1bb-0b98f0ab3f5e" />

```
df.dropna()
```
<img width="671" height="506" alt="image" src="https://github.com/user-attachments/assets/fbb11fc1-2033-47c5-96ce-2155cf2a3bb3" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="272" height="185" alt="image" src="https://github.com/user-attachments/assets/0cf7458c-770c-4330-bd67-94612342dc60" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
<img width="516" height="245" alt="image" src="https://github.com/user-attachments/assets/c928c9bc-ea33-4970-a821-3178915b61ee" />

```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1[['Height', 'Weight']] = sc.fit_transform(df1[['Height', 'Weight']])

df1.head(10)
```
<img width="563" height="452" alt="image" src="https://github.com/user-attachments/assets/25119340-15a2-4b12-bcc7-ea3b0a381a03" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="503" height="432" alt="image" src="https://github.com/user-attachments/assets/2a2514fc-7956-4e55-91a6-bdc3e7a10871" />

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="626" height="506" alt="image" src="https://github.com/user-attachments/assets/ad315721-fed7-4de1-8bca-6af5cd48671a" />

```
df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height', 'Weight']] = scaler.fit_transform(df3[['Height', 'Weight']])
df3
```
<img width="566" height="512" alt="image" src="https://github.com/user-attachments/assets/5b8ec22e-2a03-4e67-846a-9fb5dc620f99" />

```
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load the dataset to define df4
df4 = pd.read_csv('bmi.csv')

scaler = RobustScaler()
df4[['Height', 'Weight']] = scaler.fit_transform(df4[['Height', 'Weight']])
df4.head()
```
<img width="530" height="256" alt="image" src="https://github.com/user-attachments/assets/8df166de-3344-4491-878b-784ae82517f5" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="636" height="240" alt="image" src="https://github.com/user-attachments/assets/04ac2fb5-4666-444a-822f-7c5c5f3b9b4f" />

```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="242" height="101" alt="image" src="https://github.com/user-attachments/assets/9b7eb3c8-30c2-4158-9529-73120ab445b5" />

```
chi2, p,_,_  = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
<img width="390" height="62" alt="image" src="https://github.com/user-attachments/assets/5ff54c74-0241-445c-b45a-8bee2f449f5b" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data = {
'Feature1': [1, 2, 3, 4, 5],
'Feature2': ['A', 'B', 'C', 'A', 'B'],
'Feature3': [0, 1, 1, 0, 1],
'Target': [0, 1, 1, 0, 1]
}
df = pd.DataFrame (data)
X = df [['Feature1', 'Feature3']]
y = df ['Target']
selector=SelectKBest(score_func=mutual_info_classif, k = 1)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="391" height="65" alt="image" src="https://github.com/user-attachments/assets/282e529f-f4a8-44e8-bb4b-3187e11c7428" />


# RESULT:
       
