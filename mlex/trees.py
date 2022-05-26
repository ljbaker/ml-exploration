import numpy as np
import pandas as pd
from functools import partial


class dtree:
    """ A basic Decision Tree"""

    def __init__(self):
        """ Constructor """


    def read_data(self,filename):
        data = pd.read_csv('car_evaluation.csv')

        self.featureNames = data.columns[:-1].tolist()
        self.classes = data.label
        data = data[self.featureNames]

        return data, self.classes, self.featureNames


    def graph_depth(tree_dict):
        """
        Utility function for identifying tree (graph) depth
        """
        if isinstance(tree_dict, dict):

            return 1 + (max(map(dict_depth, tree_dict.values()))
                                        if tree_dict else 0)
        return 0


    def get_default_entropy(self, classes, newClasses, nData, metric='entropy'):
        """
        calculate baseline entropy and default class for a sample
        """
        information = 0
        frequency = pd.Series(newClasses).value_counts()

        for freq in frequency:
            if metric=='gini':
                information += self.calc_gini(freq/nData)
            else:
                information += self.calc_entropy(freq/nData)

        default = frequency.idxmax()

        return information, frequency, default


    def count_classes(self, classes):
        return np.unique(classes)


    def select_feature(self, information, data, featureNames, classes, forest = 0, metric='entropy'):
        """
        use entropy or gini impurity coefficient to identify best
        feature for decision to be made

        Parameters
        information
        data : list(list(int))
            data structure of feature variables
        featureNames : list(str)
            list of features to decide between
        classes : list(int)
            labels for data
        forest : int
            number of features to select at random for decision
            used when implementing decision tree as a random forest
        metric : str
            if 'entropy', uses Shannon entropy to decide most informative feature
            to make decision on
            if 'gini', uses Gini impurity

        Returns
        -------
        gain : float
            information gain
        bestFeature : str
            best feature to split decision tree
        """
        gain = np.zeros(len(featureNames))
#         featureSet = range(nFeatures)

        if forest != 0:
            np.random.shuffle(featureNames)
            featureNames = featureNames[0:forest]
        for f, feature in enumerate(featureNames):
            g = self.calc_info_gain(data,classes,feature)
            gain[f] = information - g

        bestFeature = featureNames[np.argmax(gain)]

        return gain, bestFeature


    def make_tree(self,data,classes,featureNames,maxlevel=-1,level=0,forest=0):
        """

        Make a decision tree!

        Parameters
        ----------
        data : pd.DataFrame
            n x m array of feature values for the dataset, without labels
            includes featureNames as columns
        classes : pd.Series
            n x 1 array of known target values for the dataset
        featureNames : list
            m-length list of feature names; subset of columns of data df
        maxlevel : int
            maximum depth of the tree
        level : int
            current level of the tree
            function will be used recursively to elevate level of the tree
        forest : int
            number of features to randomly exclude at each node
            if forest = 0, all features are used at every node

            e.g. forest = 2 and n_features=10, then 2 features
            will be randomly selected and 8 features excluded
            at every node.

        Returns
        -------
        tree : dict
            nested dictionary of graph structure
        """

        nData = data.shape[0]
        nFeatures = data.shape[1]

        try:
            self.featureNames
        except:
            self.featureNames = featureNames

        # List the possible classes
        newClasses = self.count_classes(classes)

        # Compute the default class (and total entropy)
        information, frequency, default = self.get_default_entropy(classes, newClasses, nData)

        # find the best feature to branch

        if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
            # Have reached an empty branch
            return default
        elif len(np.unique(classes)) == 1:
            # Only 1 class remains
            return np.unique(classes)[0]
        else:
            gain, bestFeature = self.select_feature(information,
                                                    data,
                                                    featureNames,
                                                    classes)

            tree = {bestFeature:{}}

            # recurse until complete
            # List the values that bestFeature can take
            values = []
            for idx, datapoint in data.iterrows():
                if datapoint[bestFeature] not in values:
                    values.append(datapoint[bestFeature])

            for value in values:

                newData = data[data[bestFeature] == value]
                newClasses = classes[data[bestFeature] == value]
                newNames = [f for f in featureNames if f != bestFeature]

                # Now recurse to the next level
                subtree = self.make_tree(data = newData,
                                         classes=newClasses,
                                         featureNames=newNames,
                                         maxlevel=maxlevel,
                                         level=level+1,
                                         forest=forest)

                # And on returning, add the subtree on to the tree
                tree[bestFeature][value] = subtree

            return tree


    def printTree(self,tree,name=''):
        """
        Admittedly inelegant way to plot the tree
        """
        if type(tree) == dict:
            print(name, list(tree.keys())[0])
            for item in list(tree.values())[0].keys():
                print(name, item)
                self.printTree(list(tree.values())[0][item], name + "\t")
        else:
            print(name, "\t->\t", tree)


    def calc_entropy(self, p):
        out = -p * np.log2(p)
        return np.nan_to_num(out).sum()


    def calc_gini(self, p):
        p = np.array(p)
        return 1-sum((p**2))


    def calc_info_gain(self,data,classes,feature, metric='entropy'):
        """
        Calculates the information gain based on both entropy and the Gini impurity

        Parameters
        ----------
        data : pd.DataFrame

        classes : pd.Series

        feature : str
            feature name in data.columns

        metric : str
            if 'entropy', uses Shannon entropy to decide most informative feature
            to make decision on
            if 'gini', uses Gini impurity

        Returns
        -------
        gain : float
            information gain for a decision on a given feature
        """
        gain = 0
        nData = len(data)

        values = data[feature].unique()
        featureCounts = data[feature].value_counts()
        information = np.zeros(len(featureCounts))

        # Find where those values appear in data[feature] and the corresponding class
        for v, value in enumerate(values):
            newClasses = classes.loc[data[feature]==value]
            classValues = newClasses.unique()
            classCounts = newClasses.value_counts()

            if metric == 'gini':
                information[v] = self.calc_gini(classCounts / np.sum(classCounts))
            else:
                information[v] = self.calc_entropy(classCounts / np.sum(classCounts))

            gain += featureCounts[value]/nData * information[v]

        return gain


    def graph_depth(self, tree_dict):
        """
        Utility function for identifying tree (graph) depth
        """
        if isinstance(tree_dict, dict):

            return 1 + (max(map(self.graph_depth, tree_dict.values()))
                                        if tree_dict else 0)
        return 0


    def predict(self, tree, datapoint, verbose=False):
        """
        predict a single user's class

        Parameters
        ----------
        tree : dict
            graph structure of the decision tree
        datapoint : pd.Series
            row of data from a dataframe containing an individual user's data
            with feature names on the index

        Returns
        -------
        y_hat : pd.Series
            pandas Series of predictions
            may contain Nonetype values if no graph exists for a user's data
        """
        for key, val in datapoint.iteritems():
            if key in tree.keys():
                if verbose:
                    print(f'index: {datapoint.name} {key}: {val}')
                try:
                    tree = tree[key][val]
                except:
                    tree = None
                if isinstance(tree, dict):
                    return self.predict(tree, datapoint, verbose)
                else:
                    return tree

    def predictAll(self, tree, data):
        """
        bulk predict on a pandas dataframe of values

        Parameters
        ----------
        tree : dict
            graph structure of the decision tree
        data : pd.DataFrame
            dataframe with users on the index and feature as labeled colums

        Returns
        -------
        y_hat : pd.Series
            pandas Series of predictions
            may contain Nonetype values if no graph exists for a user's data
        """
        f = partial(self.predict, tree)
        return data.apply(f, axis=1)


class randomforest(dtree):
    """
    A simple random forest algorithm, expanded from a decision tree algorithm.
    """
    def __init__(self):
        """ Constructor """
        # call super()
        super().__init__()
        self.dtree = dtree()


    def rf(self,data,classes,features,nTrees,nSamples,nFeatures,maxlevel=-1):
        """
        Random forest loop.

        Parameters
        ----------
        data : pd.DataFrame
            dataframe of values with features on columns and measurement intervals on rows
            class target labels are not on this dataframe
        classes : pd.Series
            target class labels for the data
        features : list
            list of features to train the algorithm. All features must be rows in data
        nTrees : int
            number of random trees to generate
        nSamples : int
            number of samples to randomly select per each tree in the random forest
        nFeatures : int
            number of features to randomly select per each tree in the random forest
        maxlevel : int
            maximum depth of the decision tree
            -1 leads to max depth

        Returns
        -------
        tree_list : list(dict)
            list of n graphs, where n=nTrees
        """
        tree_list = []

        for i in range(nTrees):
            print(i)
            # Compute bootstrap samples
            sample = data.sample(n=nSamples)
            sampleTarget = classes.loc[sample.index]

            tree_list.append(self.dtree.make_tree(sample,
                                                  sampleTarget,
                                                  featureNames,
                                                  maxlevel=maxlevel,
                                                  forest=nFeatures))

        return tree_list


    def rfpredict(self, tree_list, data):
        """
        predict class through majority voding of all trees generated by the random forest algorithm

        Parameters
        ----------
        tree_list : list(dict)
            list of graphs producted by iterative generation of decision trees generated by rf
        data : pd.DataFrame
            df of values with features on columns and measures per row

        Returns
        -------
        decision : pd.Series
            array of majority-vote predictions
            may contain null values for edge cases
        """
        # Majority voting
        yhats = pd.DataFrame()
        for t, tree in enumerate(tree_list):
            yhats[f'tree_{t}'] = dt.predictAll(tree, data)

        decision = yhats.mode(axis=1)
        if decision.shape[1] > 1:
            decision = decision[0]

        return decision
