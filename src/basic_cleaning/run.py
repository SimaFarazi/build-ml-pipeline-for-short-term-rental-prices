#!/usr/bin/env python
"""
Download from W&B the raw dataset and 
apply some basic data cleaning, 
exporting the result to a new artifact
"""
import argparse
import logging
import os
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact from W&B
    logger.info(f"Downloading input file {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Drop outliers
    df = pd.read_csv(artifact_local_path)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remove missing values
    df.dropna(inplace = True)

    # Save dataframe to a csv file without adding index column
    df.to_csv(args.output_artifact, index=False)

    # Log output artifact to W&B
    logger.info(f"Uploading clean file {args.output_artifact} to Weights & Biases")
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input file",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="AA brief description of this artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minumum price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="maximum price",
        required=True
    )


    args = parser.parse_args()

    go(args)
