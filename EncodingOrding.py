import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
np.set_printoptions(suppress=True)

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"



def ord(path, pathTest):
    pd.set_option('display.expand_frame_repr', False)


    df = pd.read_csv(path + ".csv")
    dfTest=pd.read_csv(pathTest + ".csv")
    df_transposed = df.T
    print(df_transposed.head(2))

    header=list(df.columns.values)
    cls='classification'
    y=df[cls]
    y=np.array(y)
    data=df.drop(columns=cls)
    data=data.T
    data=np.array(data)
    embedding = MDS(n_components=2, n_jobs=7, dissimilarity='euclidean')
    X_transformed = embedding.fit_transform(data)
    #print(X_transformed)
    p1=sns.scatterplot(X_transformed[:, 0], X_transformed[:, 1], cmap=plt.cm.get_cmap('jet', 10))

    p1.get_figure().savefig('heatmap.png')
    distance_matrix = pairwise_distances(data, metric='euclidean')
    non_zero_arr = np.extract(distance_matrix>0,distance_matrix)
    result = np.where(distance_matrix == np.amin(non_zero_arr))
    print('Tuple of arrays returned : ', result)
    featuresList=list()
    feature1=result[0][0]
    print(feature1)
    feature2=result[0][1]
    r1=distance_matrix[feature1]
    v=min([x for x in r1 if x !=0])
    distance_matrix=np.where(distance_matrix==v,  8000000, distance_matrix)
    distance_matrix=np.where(distance_matrix==0,  8000000, distance_matrix)
    r1=distance_matrix[feature1]
    r2=distance_matrix[feature2]
    v=min([x for x in r1 if x !=0])
    #print(v)
    v1=min([x for x in r2 if x !=0])
    #print(v1)
    if(v > v1):
        featuresList.append(feature1)
        featuresList.append(feature2)
    else:
        featuresList.append(feature2)
        featuresList.append(feature1)

    for i in range(8):
        last=featuresList[-1]
        #print(last)

        f =distance_matrix[last]

        f=f.tolist()
        #print(f)
        v =np.where(f == np.amin(f))
        #print('f')

        v1=v[0]

        while(v1[0] in featuresList and len(featuresList) < 10):
            value = min(f)
            distance_matrix = np.where(distance_matrix == value, 8000000, distance_matrix)
            f = distance_matrix[last]
            v = np.where(f == np.amin(f))
            #print(v)
            #print('f')
            value = min(f)
            v1 = v[0]
        featuresList.append(v1[0])
        value = min(f)
        distance_matrix = np.where(distance_matrix == value, 8000000, distance_matrix)

        #print(featuresList)

    string='feature_'

    column= [ string + str(s) for s in featuresList]
    print(column)
    column.append(cls)
    df = df.reindex(column, axis=1)
    print(df.head(2))
    dfTest=dfTest.reindex(column, axis=1)

    p=sns.heatmap(df)
    p.get_figure().savefig('heatmap2.png')
    df.to_csv(path + '_ord.csv', index=False)
    dfTest.to_csv(pathTest + '_ord.csv', index=False)
    return df, dfTest



def ordCICIDS(path, tests, pathTest, pathDatasetEncoded):
    #path = "encoded/CICDS/encoded/train_encoded"
    pd.set_option('display.expand_frame_repr', False)
    #pathTest="encoded/CICDS/encoded/test_encodedtest1_CICIDS2017OneCls"

    df = pd.read_csv(path + ".csv")
    df_transposed = df.T
    print(df_transposed.head(8))

    header=list(df.columns.values)
    cls='classification'
    y=df[cls]
    y=np.array(y)
    data=df.drop(columns=cls)
    data=data.T
    data=np.array(data)
    distance_matrix = pairwise_distances(data, metric='euclidean')
    non_zero_arr = np.extract(distance_matrix>0,distance_matrix)
    print(type(non_zero_arr))
    result = np.where(distance_matrix == np.amin(non_zero_arr))
    #print('Tuple of arrays returned : ', result[1])
    featuresList=list()
    feature1=result[0][0]
    #print(feature1)
    feature2=result[1][0]
    r1=distance_matrix[feature1]
    #print(r1)
    v=min([x for x in r1 if x !=0])
    distance_matrix=np.where(distance_matrix==v,  8000000, distance_matrix)
    distance_matrix=np.where(distance_matrix==0,  8000000, distance_matrix)
    r1=distance_matrix[feature1]
    r2=distance_matrix[feature2]
    v=min([x for x in r1 if x !=0])
    #print(v)
    v1=min([x for x in r2 if x !=0])
    #print(v1)
    if(v > v1):
        featuresList.append(feature1)
        featuresList.append(feature2)
    else:
        featuresList.append(feature2)
        featuresList.append(feature1)

    for i in range(8):
        last=featuresList[-1]
        #print(last)

        f =distance_matrix[last]

        f=f.tolist()
        #print(f)
        v =np.where(f == np.amin(f))
        #print('f')

        v1=v[0]

        while(v1[0] in featuresList and len(featuresList) < 10):
            value = min(f)
            distance_matrix = np.where(distance_matrix == value, 8000000, distance_matrix)
            f = distance_matrix[last]
            v = np.where(f == np.amin(f))
            #print(v)
            #print('f')
            value = min(f)
            v1 = v[0]
        featuresList.append(v1[0])
        value = min(f)
        distance_matrix = np.where(distance_matrix == value, 8000000, distance_matrix)

        #print(featuresList)

    string='feature_'

    column= [ string + str(s) for s in featuresList]
    print(column)
    column.append(cls)
    df = df.reindex(column, axis=1)

    df_tests=[]
    for testset, t in zip(pathTest,tests):
        dfTest=t.reindex(column, axis=1)
        df_tests.append(dfTest)
        dfTest.to_csv(pathDatasetEncoded + 'test_encoded' + testset + '_ord.csv', index=False)




    df.to_csv(path + '_ord.csv', index=False)
    return df, df_tests







