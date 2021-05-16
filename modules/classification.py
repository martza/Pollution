from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# I need to be able to read the time
x_new = {'year':2020, 'month':12, 'day':25, 'hour':17, 'Longitude':50.8864501, 'Latitude':4.7038946,
                'Altitude':0.00}
# I will need to infer Weekday and season. I need to infer AirQuality stastin type and airquality station area

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
                            ['season'])])# pass all categorical variables

    # Use one-hot encoding for categorical variables

    print('Original features:\n',list(x.columns),'\n')
    ct.fit(x)                                                                   # Get dummies for categorical variables
    x_dummies = ct.transform(x)
    x_dummies_df = pd.DataFrame(data = x_dummies, columns = ct.get_feature_names())# Transform to a dataframe
    x.reset_index(drop = True, inplace = True)
    x.drop(columns = ['season'], inplace = True)                                                      # Get numerical variables
    x = x.merge(x_dummies_df, left_index = True, right_index = True)            # Merge numerical variables with cat. dummies
    print('Features after get_dummies:\n',list(x.columns),'\n')
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

def GBC(x_train,x_test,y_train,y_test):
    gbrt = GradientBoostingClassifier(random_state = 0)
    gbrt.fit(x_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(gbrt.score(x_train,y_train)))
    print('Accuracy on the test set: {:.3f}'.format(gbrt.score(x_test,y_test)))
    y_new_pred = gbrt.predict(x_new)
    print(y_new_pred)
    plt.bar(x = x_train.columns, height=gbrt.feature_importances_ )
    plt.xticks(rotation = 90)
    plt.show()



if __name__ == '__main__':
    import datetime
    data = pd.read_csv('..\\data\\prediction_SO2.csv')
    data.columns
    data.shape
    data.Altitude.unique()
    data = data.loc[data.Altitude <10]
    data.shape
    data.drop(['Altitude','year','season'],axis = 1, inplace=True)
    # point for prediction
    coord = [4.7038946, 50.8864501]
    date = datetime.date.today()

    new_data = {'Longitude':coord[0], 'Latitude':coord[1],
                'month':date.month, 'day':date.day,
                'weekday':date.weekday(), 'hour':1 }
    new = pd.DataFrame(new_data, index = [0])


    # vectors
    x = data.drop('Concentration', axis=1)
    y = data.Concentration

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                        random_state = 0)

    print('x_train_shape : ', x_train.shape)
    print('y_train_shape : ', y_train.shape)
    print('x_test_shape : ', x_test.shape)
    print('y_test_shape : ', y_test.shape)

    # Get dummies
    x_train = get_transform(x_train)
    x_test = get_transform(x_test)                                              # look what _self argument does


    max_depth_test = np.array([[0,0,0,0]])
    d = 1
    diff = 1
    eps = 1e-8
    test_err = 1

    while (abs(diff)>eps):

        print(d)
        # f = 0
        # max_depth_test = np.append(max_depth_test, [[d,0,0,0]], axis = 0)
        # f >=1
        test_err_f = 1
        temp = np.array([[0,test_err_f,1]])
        for f in np.arange(1,len(x_train.columns)+1):
            print(f)
            tree = DecisionTreeRegressor(criterion='mse', splitter='best',
                        max_depth = d, min_samples_split=2,
                        min_samples_leaf=1, max_features=f, random_state=0)
            tree.fit(x_train, y_train)
            diff_f = abs(tree.score(x_test,y_test)-test_err_f)
            test_err_f = tree.score(x_test,y_test)
            test_train_f = tree.score(x_train,y_train)
            temp = np.append(temp, [[f, test_err_f, test_train_f]], axis = 0)
            if(diff_f < eps):
                element = [[d,f,tree.score(x_test,y_test),tree.score(x_train,y_train)]]
                #print(element)
                max_depth_test = np.append(max_depth_test,element,axis = 0)
                break
            elif(f==len(x_train.columns)):
                idx = np.amin(temp[:,2], axis = 0)
                element = [[d,temp[idx,0], temp[idx,1], temp[idx,2]]]
                max_depth_test = np.append(max_depth_test,element,axis = 0)

        diff = abs(test_err - max_depth_test[-1,2])
        test_err = max_depth_test[-1,2]
        print(diff>eps)
        print(diff)
        d+=1
print(max_depth_test)

max_depth_test[-1, 2]
    a = np.array([[0,0,0,0]])

    type(a)
    for i in np.arange(1, 3):
        for j in np.arange(1,3):
            b = [[i,j, j+1, i]]
            a = np.append(a, b, axis = 0)

    print(a)
    min(a[:,2])
    a[np.amin(a[:,2], axis = 0),:]
    a[1,2]

fig,ax = plt.subplots()
ax.plot(max_depth_test[:,0],max_depth_test[:,1])
ax.plot(max_depth_test[:,0],max_depth_test[:,2])
ax.set(title = 'convergence of max_depth parameter', ylabel= 'score metric', xlabel = 'max_depth')
plt.show()


tree = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth = 15,
                                 min_samples_split=2, min_samples_leaf=1,
                                 max_features='auto', random_state=0)
tree.fit(x_train, y_train)

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = 'tree.dot', feature_names = x_train.columns,
                impurity = False, filled = True)

import graphviz

with open('tree.dot') as f:
    dot_graph = f.read()

display(graphviz.Source(dot_graph))

    print('The depth of the tree is: {:.3f}'.format(tree.get_depth()))
    print('The number of leaves in the decision tree is: {:.3f}'.format(tree.get_n_leaves()))
    print('Accuracy on training set: {:.3f}'.format(tree.score(x_train,y_train)))
    print('Accuracy on the test set: {:.3f}'.format(tree.score(x_test,y_test)))

    # plot feature importances
plt.bar(x = x_train.columns, height=tree.feature_importances_ )
plt.xticks(rotation = 90)
plt.show()

data.columns
data.month.unique()
#plt.savefig('output\tree_feature_importances.png')


#    y_train.value_counts()                                                     For requesting user input for prediction
#    type(x_train.Altitude.value_counts())
#    x_train.Altitude.value_counts().sort_index()

#    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)
#    plot_tree(tree, feature_names = x_dummies.columns, class_names = ['Good','Fair','Moderate','Poor'])
#    fig.savefig('imagename.png')
