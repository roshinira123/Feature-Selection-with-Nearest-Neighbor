import math 
import copy 


#def loadData 
def load_data(filename):
    data = []
    try:
        with open(filename, "r") as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                if values: 
                    data.append(values)

        if not data: #checks if the dataset is empty or not formatted correctly
            print("Error: The file is empty or not formatted correctly.")
            return None, None

      
        y = [row[0] for row in data]  # Class labels -1st
        x = [row[1:] for row in data]  # Feature values -2nd

        print("Successfully loaded dataset!")
        return x, y
    
    #other exceptions 
    except FileNotFoundError:
        print("The file was not found.")
        return None, None
    except ValueError:
        print("Could not convert data")
        return None, None

#def euclideanDist for nearest neighbor 
def euclideanDist(x1, x2): #works
    distance = 0
    for i in range(len(x1)):
        difference = x1[i] - x2[i]  
        squared = difference ** 2  
        distance += squared
    return math.sqrt(distance) 

#def leaveoneoutcrossvalidation
def loocv(X, y, features = None): 
    correct = 0 #stores correct predict
    total = len(X) #total instances

    for i in range(total):
        #if no feature is selected then use all the features in the set
        if features is None:
            trainX = []
            #build training set w/o test val
            for j, row in enumerate(X):
                if j != i:
                    trainX.append(row)
                    #use curr data point as test val
            test_point = X[i]
        else:
            trainX = []
            #build set only w/ selected features
            for j, row in enumerate(X):
                if j != i:
                    selected_features = [row[f] for f in features]
                    trainX.append(selected_features)
            test_point = [X[i][f] for f in features]
        #build label set off of curr data point
        trainY = []
        for j in range(total):
            if j != i:
                trainY.append(y[j])
        label = y[i]
        #predict the label using nearest neighbor
        predicted_label = nearestNeighbor(trainX, trainY, test_point)
        if predicted_label == label:
            correct += 1 #increment correct if predicted label is correct
    return correct / total

#def nearest neighbor
def nearestNeighbor(trainX, trainY, testVal): #works
    minDist = float("inf") #stores min dist
    NN = 0 #gets the nearest neighbor
    
    for i in range(len(trainX)):
        dist = euclideanDist(trainX[i], testVal) #Computes the dist
        #compares new value with old val and assigns closet neighbor
        if dist < minDist:  
            minDist = dist
            NN = trainY[i]

    return NN 

#def forward selection 
def forwardSelection(x, y): 

    currFeatures = []
    num_features = len(x[0])
    bestFeatures = []
    bestOverallaccuracy = 0


    print("\nBeginning search.")

    for i in range(num_features): #iterates thru all features 
        feature_to_add = None #tracks whther that feature is best or not
        best_accuracy = 0 #stores best accuracy

        for k in range(num_features):

            if k not in currFeatures: #checks if feature is already in the set
                #does loocv with curr features + new feature
                accuracy = loocv(x, y, currFeatures + [k])

                print("\t"+ f"Using feature(s) {{{', '.join(str(f+1) for f in currFeatures + [k])}}} accuracy is {accuracy * 100:.1f}%")
                if accuracy > best_accuracy: #updates for best accuracy 
                    best_accuracy = accuracy
                    feature_to_add = k
        #if feature was found add to current feature set
        if feature_to_add is not None:
            currFeatures.append(feature_to_add)
            print(f"Feature set {{{', '.join(str(f+1) for f in currFeatures)}}} was best, accuracy is {best_accuracy * 100:.1f}%")
            if best_accuracy > bestOverallaccuracy:
                bestOverallaccuracy = best_accuracy
                bestFeatures = list(currFeatures)

    print(f"\nFinished search!! The best feature subset is {{{', '.join(str(f+1) for f in bestFeatures)}}}, which has an accuracy of {bestOverallaccuracy * 100:.1f}%") #prints out the best overall accuracy


#def backwards elimination



#def main
def main():
    print("Welcome to Roshini's Feature Selection Algorithm.")
    # Load dataset
    file =  input("Type in the name of the file to test: ").strip()
    x, y = load_data(file)

    #splits values into x and y for features and instances
    if x is None or y is None:
        exit()
    num_features = len(x[0])
    num_instances = len(x)
    print(f"\nThis dataset has {num_features} features with {num_instances} instances.")

    # train_X = [[1, 2], [3, 4], [5, 6]]
    # train_y = [1, 1, 0]
    # test_point = [2, 3]
    # print("Predicted Label:", nearestNeighbor(train_X, train_y, test_point)) #testing NN

   
    #forwardSelection(x, y) #works - testing for forward selection



if __name__ == '__main__':
	main()