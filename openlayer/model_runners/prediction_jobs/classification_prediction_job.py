"""Script that runs a classification prediction job.

This file will get copied into the model package when the user uploads a model.

The input and output are written to csv files in
the path specified by the --input and --output flags.

Example usage:
    python classification_prediction_job.py --input /path/to/input.csv --output /path/to/output.csv
"""
import argparse
import logging

import pandas as pd
import prediction_interface

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Parse args
    logger.debug("Parsing args to run the prediction job...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input_data_file_path")
    parser.add_argument("--output", action="store", dest="output_data_file_path")
    args = parser.parse_args()

    # Load input data
    logger.debug("Loading input data...")
    input_data = pd.read_csv(args.input_data_file_path)

    # Load model module
    logger.debug("Loading model...")
    ml_model = prediction_interface.load_model()

    # Run model
    logger.debug("Running model...")
    output_data = pd.DataFrame(
        {"predictions": ml_model.predict_proba(input_data).tolist()}
    )

    # Save output data
    logger.debug("Saving output data...")
    output_data.to_csv(args.output_data_file_path, index=False)
