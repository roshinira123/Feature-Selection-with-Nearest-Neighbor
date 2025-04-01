# Feature Selection with Nearest Neighbor Classifier

This project implements a Nearest Neighbor classifier with feature selection using Forward Selection and Backward Elimination. The goal is to identify the most relevant features that improve classification accuracy. The implementation is done in pure Python, utilizing the `copy` and `math` libraries without relying on external dependencies like NumPy. This project was created for the CS170 class (Introduction to Artificial Intelligence) at UC Riverside, which was taught by Eamonn Keogh in Winter 2025.

## Features
- **Nearest Neighbor Classifier**: Implements a simple nearest neighbor algorithm for classification.
- **Feature Selection**:
  - **Forward Selection**: Iteratively adds the best feature that improves classification performance.
  - **Backward Elimination**: Starts with all features and iteratively removes the least important one.
- **Customizable Input**: Prompts users for file input and algorithm selection.
- **Step-by-Step Output**: Displays the detailed progression of feature selection.

## Installation
No additional libraries are required. Ensure you have Python installed on your system.

```sh
python3 main.py
```

## Usage
1. Run the script:
   ```sh
   python3 main.py
   ```
2. Provide the required inputs when prompted:
   - Enter the dataset filename.
   - Choose Forward Selection or Backward Elimination.
3. View the step-by-step output of the feature selection process and the final selected feature set.
