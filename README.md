# Dissertation

A Python-based ETL (Extract, Transform, Load) project for processing and analyzing vulnerability datasets.

## Project Overview

This project provides a pipeline for extracting, transforming, and merging vulnerability data from NVD CVE JSON datasets into structured CSV formats for analysis.

## Project Structure

```
├── 01_extract_year.py           # Extract data from JSON datasets
├── 02_transform_year.py         # Transform extracted data
├── 03_merge_and_build_series.py # Merge and aggregate data
├── Datasets/
│   ├── JSON_datasets/           # Input JSON files (NVD CVE format)
│   └── Extracted_Data/          # Processed CSV outputs
└── README.md
```

## Usage

### 1. Extract Year Data
```bash
python 01_extract_year.py --input Datasets/JSON_datasets/nvdcve-2.0-2022.json --output Datasets/Extracted_Data/extracted_2022.csv
```

### 2. Transform Year Data
```bash
python 02_transform_year.py --input Datasets/Extracted_Data/extracted_2019.csv --output Datasets/Extracted_Data/transformed_2019.csv
```

### 3. Merge and Build Time Series
```bash
python 03_merge_and_build_series.py --input_dir Datasets/Extracted_Data --output_dir Datasets/Extracted_Data
```

## Requirements

- Python 3.x
- Required dependencies (see requirements.txt for details)

## License

This project is part of a dissertation research initiative.
