import pandas as pd

def main():
    # Load the dataset
    df = pd.read_csv('SRM_assignment_survey_responses.csv')
    
    # Display the first few rows of the dataset
    print(df.head())
    
    # Perform some basic analysis
    print("Summary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
    