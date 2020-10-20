from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# list of variables for machine learning
ml_variables = ['Concentration', 'year', 'month', 'day', 'hour', 'weekday',
                'season', 'Longitude', 'Latitude', 'Altitude',
                'AirQualityStationType', 'AirQualityStationArea']

def drawPairplot(data):
    plt1 = sns.pairplot(data, hue = 'AQindex', hue_order=['Good','Fair','Moderate','Poor','Extremely Poor'],
                        palette=None,vars=None, x_vars=['hour','weekday','month','year','season'],
                        y_vars='Concentration', kind='scatter', diag_kind='auto',
                        markers=None, height=5, aspect=1, corner=False, dropna=False,
                        plot_kws=None, diag_kws=None, grid_kws=None, size=None)

    plt1.savefig('output\scatter_matrix.png')
    plt.close()


def drawStripplot(data):
    plt2 = sns.stripplot(x='season', y='Concentration', hue='AQindex', data=data,
                hue_order=['Good','Fair','Moderate','Poor','Extremely Poor'])
    plt2.figure.savefig('output\stripplot.png')
    plt.close()


def get_transform(x):

    ct = ColumnTransformer([('onehot', OneHotEncoder(sparse = False),
                            ['AirQualityStationType','AirQualityStationArea'])])# pass all categorical variables

    # Use one-hot encoding for categorical variables

    print('Original features:\n',list(x.columns),'\n')
    ct.fit(x)                                                                   # Get dummies for categorical variables
    x_dummies = ct.transform(x)
    x_dummies_df = pd.DataFrame(data = x_dummies, columns = ct.get_feature_names())# Transform to a dataframe
    x.reset_index(drop = True, inplace = True)
    x.drop(columns = ['AirQualityStationType', 'AirQualityStationArea'],
           inplace = True)                                                      # Get numerical variables
    x = x.merge(x_dummies_df, left_index = True, right_index = True)            # Merge numerical variables with cat. dummies
    print('Features after get_dummies:\n',list(x.shape),'\n')
    return x

def tree_regression(x_train, y_train, x_test, y_test):

    tree = DecisionTreeRegressor(criterion='mse', splitter='best',
                                 min_samples_split=0.5, min_samples_leaf=0.05,
                                 max_features='auto', random_state=0)

    tree.fit(x_train, y_train)

    print('The depth of the tree is: {:.3f}'.format(tree.get_depth()))
    print('The number of leaves in the decision tree is: {:.3f}'.format(tree.get_n_leaves()))
    print('Accuracy on training set: {:.3f}'.format(tree.score(x_train,y_train)))
    print('Accuracy on the test set: {:.3f}'.format(tree.score(x_test,y_test)))

    # plot feature importances
    plt.bar(x = x_train.columns, height=tree.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.savefig('output\tree_feature_importances.png')

    # Prediction take new region with the following characteristics
 #   x_new = pd.DataFrame(data = [[2020,10,8,1,117.0,1,0,0,0,0,0,0,0,1]], columns=x_test.columns)
 #   y_new_pred = tree.predict(x_new)
 #   print(y_new_pred)

def forest_regression(x_train, y_train, x_test, y_test):

    forest = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
    forest.fit(x_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(forest.score(x_train,y_train)))
    print('Accuracy on the test set: {:.3f}'.format(forest.score(x_test,y_test)))
    y_new_pred = forest.predict(x_new)
    print(y_new_pred)
    plt.bar(x = x_train.columns, height=forest.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()

gbrt = GradientBoostingClassifier(random_state = 0)
gbrt.fit(x_train, y_train)
print('Accuracy on training set: {:.3f}'.format(forest.score(x_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(forest.score(x_test,y_test)))
y_new_pred = gbrt.predict(x_new)
print(y_new_pred)
    plt.bar(x = x_train.columns, height=gbrt.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()



if __name__ == '__main__':
    data = pd.read_csv('data\AT__1_2017_2020.csv')
    data = data[ml_variables]

    # vectors
    x = data.drop('Concentration')
    y = data.Concentration

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y,
                                        random_state = 0)

    print('x_train_shape : ', x_train.shape)
    print('y_train_shape : ', y_train.shape)
    print('x_test_shape : ', x_test.shape)
    print('y_test_shape : ', y_test.shape)

    x_train = get_transform(x_train)
    x_test = get_transform(x_test)                                              # look what _self argument does

#    y_train.value_counts()                                                     For requesting user input for prediction
#    type(x_train.Altitude.value_counts())
#    x_train.Altitude.value_counts().sort_index()

#    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)
#    plot_tree(tree, feature_names = x_dummies.columns, class_names = ['Good','Fair','Moderate','Poor'])
#    fig.savefig('imagename.png')
