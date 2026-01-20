"""
Script to test and demonstrate Data Ingestion Component
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from Networksecurity.Components.data_ingestion import DataIngestion, DataIngestionConfig

def main():
    """
    Main function to test data ingestion
    """
    try:
        print("\n" + "="*60)
        print("TESTING DATA INGESTION COMPONENT")
        print("="*60 + "\n")
        
        # Option 1: Use default configuration
        print("Creating Data Ingestion with default config...")
        data_ingestion = DataIngestion()
        
        # Option 2: Use custom configuration (commented out)
        # custom_config = DataIngestionConfig(
        #     raw_data_path='custom_artifacts/raw_data.csv',
        #     train_data_path='custom_artifacts/train.csv',
        #     test_data_path='custom_artifacts/test.csv',
        #     test_size=0.25,
        #     random_state=123
        # )
        # data_ingestion = DataIngestion(config=custom_config)
        
        # Run data ingestion process
        print("\nStarting data ingestion process...\n")
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Display results
        print("\n" + "="*60)
        print("DATA INGESTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n✓ Train Data saved at: {train_path}")
        print(f"✓ Test Data saved at: {test_path}")
        print(f"✓ Raw Data saved at: {data_ingestion.config.raw_data_path}")
        
        # Load and display basic info about saved data
        import pandas as pd
        
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"\nTrain Set:")
        print(f"  - Shape: {train_df.shape}")
        print(f"  - Churn distribution:\n{train_df['Churn'].value_counts()}")
        
        print(f"\nTest Set:")
        print(f"  - Shape: {test_df.shape}")
        print(f"  - Churn distribution:\n{test_df['Churn'].value_counts()}")
        
        print("\n✓ All artifacts created successfully!")
        print("\nYou can now use these files for model training.\n")
        
        return train_path, test_path
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
