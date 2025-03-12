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

        if not data: #if the dataset is empty or not formatted correctly
            print("Error: The file is empty or not formatted correctly.")
            return None, None

        # Extract class labels (first column) and features (remaining columns)
        y = [row[0] for row in data]  # Class labels
        x = [row[1:] for row in data]  # Feature values

        print("Successfully loaded dataset!")
        return x, y
    
    #other exceptions 
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None, None
    except ValueError:
        print("Error: Could not convert data to float.")
        return None, None

#def euclideanDist for nearest neighbor 

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





if __name__ == '__main__':
	main()