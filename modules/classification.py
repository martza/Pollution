from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def drawPairplot(data):
    plt1 = sns.pairplot(data, hue = 'AQindex', hue_order=['Good','Fair','Moderate','Poor','Extremely Poor'],
                        palette=None,vars=None, x_vars=['hour','weekday','month','year','season'],
                        y_vars='Concentration', kind='scatter', diag_kind='auto',
                        markers=None, height=5, aspect=1, corner=False, dropna=False,
                        plot_kws=None, diag_kws=None, grid_kws=None, size=None)

    plt1.savefig('scatter_matrix.png')
    plt.close()

def drawStripplot(data):
    plt2 = sns.stripplot(x='season', y='Concentration', hue='AQindex', data=data,
                hue_order=['Good','Fair','Moderate','Poor','Extremely Poor'])
    plt2.figure.savefig('stripplot.png')
    plt.close()

if __name__ == '__main__':
    data = pd.read_csv('working_dataset.csv')

    temporal_data = data[['year','month','hour','weekday','season','Concentration','AQindex']]
    # optional plots
    pairplot = False
    if pairplot : drawPairplot(temporal_data)

    stripplot = False
    if stripplot : drawStripplot(temporal_data)

    # vectors
    x = data[['year', 'month', 'hour', 'weekday', 'Altitude',
                         'AirQualityStationType', 'AirQualityStationArea']]
    y = data.AQindex

    # Transformations to categorical variables
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    ct = ColumnTransformer([('onehot', OneHotEncoder(sparse = False),
                            ['AirQualityStationType','AirQualityStationArea'])])

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y,
                                        random_state = 0)
    print('x_train_shape : ', x_train.shape)
    print('y_train_shape : ', y_train.shape)
    print('x_test_shape : ', x_test.shape)
    print('y_test_shape : ', y_test.shape)

    # Use one-hot encoding for categorical variables
    print('Original features:\n',list(x_train.columns),'\n')
    ct.fit(x_train)
    x_train_dummies = ct.transform(x_train)
    x_train_dummies_df = pd.DataFrame(data = x_train_dummies, columns = ct.get_feature_names())
    x_train.reset_index(drop = True, inplace = True)
    x_train.drop(columns = ['AirQualityStationType', 'AirQualityStationArea'], inplace = True)
    x_train = x_train.merge(x_train_dummies_df, left_index = True, right_index = True)

    #pd.get_dummies(x_train)

    print('Features after get_dummies:\n',list(x_train_dummies.shape),'\n')
    #x_test_dummies = pd.get_dummies(x_test)
    x_test_dummies = ct.transform(x_test)

    x_test_dummies_df = pd.DataFrame(data = x_test_dummies, columns = ct.get_feature_names())
    x_test_dummies_df.head()
    x_test.reset_index(drop = True, inplace = True)
    x_test.drop(columns = ['AirQualityStationType', 'AirQualityStationArea'], inplace = True)
    x_test = x_test.merge(x_test_dummies_df, left_index = True, right_index = True)


    y_train.value_counts()
    type(x_train.Altitude.value_counts())
    x_train.Altitude.value_counts().sort_index()

    tree = DecisionTreeClassifier(max_depth = 10, random_state = 0)
    tree.fit(x_train, y_train)

    print('Accuracy on training set: {:.3f}'.format(tree.score(x_train,y_train)))
    print('Accuracy on the test set: {:.3f}'.format(tree.score(x_test,y_test)))

    tree.feature_importances_
    x_train.shape[1]
    plt.bar(x = x_train.columns, height=tree.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()

    y_pred = tree.predict(x_test)
    tree.score(x_test, y_test)
    x_test.columns
    x_test.dtypes
# Prediction take new region with the following characteristics
x_test.columns
x_new = pd.DataFrame(data = [[2020,10,8,1,117.0,1,0,0,0,0,0,0,0,1]], columns=x_test.columns)
y_new_pred = tree.predict(x_new)
print(y_new_pred)

# Use random forests
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
forest.fit(x_train, y_train)
print('Accuracy on training set: {:.3f}'.format(forest.score(x_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(forest.score(x_test,y_test)))
y_new_pred = forest.predict(x_new)
print(y_new_pred)
    plt.bar(x = x_train.columns, height=forest.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()

from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state = 0)
gbrt.fit(x_train, y_train)
print('Accuracy on training set: {:.3f}'.format(forest.score(x_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(forest.score(x_test,y_test)))
y_new_pred = gbrt.predict(x_new)
print(y_new_pred)
    plt.bar(x = x_train.columns, height=gbrt.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()


#    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)
#    plot_tree(tree, feature_names = x_dummies.columns, class_names = ['Good','Fair','Moderate','Poor'])
#    fig.savefig('imagename.png')
