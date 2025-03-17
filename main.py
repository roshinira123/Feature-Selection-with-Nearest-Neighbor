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




if __name__ == '__main__':
	main()