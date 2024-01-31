import pandas as pd
path='Movie Interests.csv'
movie_info=pd.read_csv(path)
'''Age  Gender   Interest
0     8       1  Animation
1    11       1  Animation
2    12       1  Animation
3    16       1     Action
4    18       1     Action
5    19       1     Action
6    23       1      Drama
7    26       1      Drama
8    27       1      Drama
9     7       0  Animation
10    9       0  Animation
11   10       0  Animation
12   26       0     Action
13   27       0     Action
14   30       0     Action
15   31       0      Drama
16   34       0      Drama
17   35       0      Drama'''


input_details=movie_info.drop(columns=['Interest'])
'''    Age  Gender
0     8       1
1    11       1
2    12       1
3    16       1
4    18       1
5    19       1
6    23       1
7    26       1
8    27       1
9     7       0
10    9       0
11   10       0
12   26       0
13   27       0
14   30       0
15   31       0
16   34       0
17   35       0'''

output_details=movie_info['Interest']
'''0     Animation
1     Animation
2     Animation
3        Action
4        Action
5        Action
6         Drama
7         Drama
8         Drama
9     Animation
10    Animation
11    Animation
12       Action
13       Action
14       Action
15        Drama
16        Drama
17        Drama
Name: Interest, dtype: object'''

#create AI model
from sklearn.tree import DecisionTreeClassifier
movie_model=DecisionTreeClassifier()
movie_model.fit(input_details,output_details)
result=movie_model.predict([[9,1],[33,0]])
# output :array(['Animation', 'Drama'], dtype=object)
