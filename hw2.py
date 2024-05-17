import queue
import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    #Extract the lable column of the current data.
    label_column = data[:, -1]
    
    #Calculate the number of instances per class
    _, amount = np.unique(label_column, return_counts=True)
    numOfInstances = len(label_column)
    
    #Calculate gini impurity
    probability = (amount/numOfInstances)
    sum = np.sum(probability**2)
    gini = 1 - sum    
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    #Extract the lable column of the current data.
    label_column = data[:, -1]
    
    #Calculate the number of instances per class
    _, amount = np.unique(label_column, return_counts=True)
    numOfInstances = len(label_column)
    
    #Calculate gini impurity
    probability = (amount/numOfInstances)
    entropy = np.sum(probability*np.log2(probability))*-1
    return entropy

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.
        Returns:
        - pred: the prediction of the node
        """
        pred = None
        #Extract the lable column of the current data.
        label_column = self.data[:, -1]
    
        #Find the majority class within the node
        uniqueElements, amount = np.unique(label_column, return_counts=True)
        max_index = np.argmax(amount)
        pred = uniqueElements[max_index]
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        # Calculate number of samples in the current node
        node_samples =len(self.data)
        goodness, _ = self.goodness_of_split(self.feature)
        # Calculate importance
        self.feature_importance = (node_samples/n_total_sample)*goodness
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        # If gain ratio is true, we must use entropy
        if self.gain_ratio == True:
            self.impurity_func = calc_entropy
         
        # Calculate current nodes impurity   
        node_impurity = self.impurity_func(self.data)
        
        # Split data by feature values
        feature_column = self.data[:, feature]
        uniqueElements, _ = np.unique(feature_column, return_counts=True)
        sum_child_impurity = 0
        split_info = 0
        
        # For each unique value of the feature insert into dictionairy 
        for element in uniqueElements:
            filter = (self.data[:, feature] == element)
            groups[element] = self.data[filter,:] 
            # Calculate impurity of the subdata
            weightedValue = (len(groups[element])/len(self.data))
            sum_child_impurity += weightedValue*self.impurity_func(groups[element])
            split_info += weightedValue*np.log2(weightedValue)
        
        # Calculate goodness of split
        goodness = node_impurity - sum_child_impurity
        
        # Calcualte gain ratio if needed
        if self.gain_ratio == True:
            if split_info == 0:
                return 0, groups
            goodness = goodness*((-1)*split_info)
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check we have not exceeded the maximum depth.
        if self.depth == self.max_depth:
            self.terminal = True
            return

        # Find subdata for best feature according to goodness of split.
        max_goodness = 0
        split_values = None
        features = [num for num in range (len(self.data[0])-1)]
        for feature in features:
            goodness,values = self.goodness_of_split(feature)
            if (goodness > max_goodness):
                max_goodness = goodness
                split_values = values
                    
        # If the chi value is larger than in the chi table allow splitting.
        if len(split_values) > 1 and self.calculate_chi(split_values):
            for key, group in split_values.items():
                child = DecisionNode(group, depth=self.depth+1, impurity_func=self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                child.calc_feature_importance(len(self.data))  # Calculate feature importance for the child node
                self.add_child(child, key)
            return

        else:
            # Mark that this is a leaf so no more splitting occurs.
            self.terminal = True
            return

        
        
    def calculate_chi(self, split_values):
        """
        Checks the current nodes chi value for splitting

        Input:
        - node: the tree itself
        - split_values: the sub data of the node according to the best feature

        Output: Boolean value that represents if the calculated chi is larger than the chi in the chi table.
        """
        # If chi is 1 allow splitting.
        if self.chi == 1:
            return True

        # Use the chi value formula for calculating.
        chi_square = 0
        
        # Extract the label column of all the data and save the amount of each label.
        label_column = self.data[:, -1]
        size = len(label_column)
        
        uniqueElements, amount = np.unique(label_column, return_counts=True)
        data_label_dict = dict(zip(uniqueElements, amount))
        
        # Iterate over the split values and calculate the chi square value.
        for _, subdata in split_values.items():
            sub_size = len(subdata)
            
            # Count the occurrences of each label in the subdata.
            label, count = np.unique(subdata[:, -1], return_counts=True)
            subdata_label_dict = dict(zip(label, count))
            
            for element, amount in data_label_dict.items():
                # If the label doesn't exist in the subdata, set observed count to 0.
                if(element not in subdata_label_dict.keys()):
                    observed = 0
                
                else: 
                    observed = subdata_label_dict.get(element)
            
                expected = sub_size * (amount / size)
                
                # According to the formula.
                chi_square = chi_square + (((observed-expected)**2) / expected)
        
        deg_of_freedom = len(split_values) - 1
        
        # Extract the chi value from the table.
        chi_val_from_table = chi_table[deg_of_freedom][self.chi]
        
        return chi_square >= chi_val_from_table


    def tree_depth(node):
        """
        Calculates the depth of the tree.
        """
        if node.terminal:
            return node.depth
        else:
            depth = 0
            for child in node.children:
                depth = max(depth, child.tree_depth())
            return depth
                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        self.root = DecisionNode(self.data, self.impurity_func, depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        queue_of_nodes = queue.Queue()
        queue_of_nodes.put(self.root)
    
        while not queue_of_nodes.empty():
            node = queue_of_nodes.get()
            
            # Find best feature for splitting
            max_goodness = 0
            best_feature = None 
            features = [num for num in range (len(self.data[0])-1)]
            for feature in features:
                goodness,_ = node.goodness_of_split(feature)
                if (goodness > max_goodness):
                    max_goodness = goodness
                    best_feature = feature
            
            # Update the best feature
            node.feature = best_feature
            
            # If the leafs are pure, goodness of split is 0, or there is not feature to split by, stop splitting this branch
            arrOfUniqueVals = np.unique(node.data)
            if len(arrOfUniqueVals) == 1 or max_goodness == 0 or node.feature == None:
                node.terminal = True
                continue
                
            # Split node by best feature
            node.split()
            for child in node.children:
                queue_of_nodes.put(child)

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        current_node = self.root
        found_child = True
        # Traverse the tree until reaching a leaf
        while found_child == True and not current_node.terminal:
            current_feature = current_node.feature
            current_instance_value = instance[current_feature]
            
            # Find the child node that matches the instances value for the feature
            found_child = False
            for i, child_value in enumerate(current_node.children_values):
                if current_instance_value == child_value:
                    current_node = current_node.children[i]
                    found_child = True
                    break
            
            if not found_child:
                break
        pred = current_node.pred
        return pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct_pred_count = 0
        total_instances = len(dataset)
        
        # Iterate over each instance and count the number of correct
        for row_instance in dataset:
            prediction = self.predict(row_instance)
            if prediction == row_instance[-1]:
                correct_pred_count += 1
        accuracy = (correct_pred_count/total_instances)*100
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training_data = tree.calc_accuracy(X_train)
        validation_data = tree.calc_accuracy(X_validation)
        training.append(training_data)
        validation.append(validation_data)

    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []
    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(X_train, calc_entropy, chi=chi_val, gain_ratio=True)
        tree.build_tree()
        training_data = tree.calc_accuracy(X_train)
        validation_data = tree.calc_accuracy(X_test)
        chi_training_acc.append(training_data)
        chi_validation_acc.append(validation_data)
        depth.append(tree.root.tree_depth())
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    n_nodes = 0
    if node is not None:
        n_nodes += 1
        for child in node.children:
            n_nodes += count_nodes(child)

    return n_nodes






