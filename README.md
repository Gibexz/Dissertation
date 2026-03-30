# Dissertation


# Sample use of python ETL codes
python 01_extract_year.py --input Datasets/JSON_datasets/nvdcve-2.0-2022.json --output Datasets/Extracted_Data/extracted_2022.csv
python 02_transform_year.py --input Datasets/Extracted_Data/extracted_2019.csv --output Datasets/Extracted_Data/transformed_2019.csv
python 03_merge_and_build_series.py --input_dir Datasets/Extracted_Data --output_dir Datasets/Extracted_Data
