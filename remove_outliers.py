import argparse
from collections import Counter
from colorama import Fore
import contextlib
import csv
import os
import progressbar
import sys
from typing import Dict, Text
import wave


FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]


def is_input_longer(duration_ms: float, transcript_len: int) -> bool:
    """Check if inputs are longer than outputs.

    Parameters
    ----------
    duration_ms: float
        duration of the audio (in milliseconds)

    transcript_len: int
        length of the transcript (number of characters)

    Returns
    ----------
    is_input_longer: bool
        if inputs is longer than outputs
    """
    return duration_ms >= (transcript_len * 20) + 12


def remove_outliers(filename: Text, clips_dir: Text) -> Dict:
    """Find and remove outliers.

    Parameters
    ----------
    filename: str
        path to the csv file to process

    clips_dir: str
        path to the clips directory
    """
    rows = list()
    counter = get_counter()
    print("Cleaning CSV file: ", filename)
    with open(filename) as read_file:
        csv_reader = csv.DictReader(read_file)
        for row in progressbar.progressbar(list(csv_reader)):
            counter["all"] += 1
            audio_file = row["wav_filename"]
            transcript = row["transcript"]
            with contextlib.closing(wave.open(clips_dir + "/" + audio_file)) as file:
                frames = file.getnframes()
                rate = file.getframerate()
                duration_ms = (frames / float(rate)) * 1000
                transcript_len = len(transcript)
                if is_input_longer(duration_ms, transcript_len):
                    rows.append(row)

    if (counter["all"] - len(rows)) > 0:
        counter["failed"] = counter["all"] - len(rows)

    print("Saving new cleaned CSV file to: ", filename)
    with open(filename, "w") as write_file:
        csv_writer = csv.DictWriter(write_file, fieldnames=FIELDNAMES)
        csv_writer.writeheader()
        for row in progressbar.progressbar(rows):
            csv_writer.writerow(row)

    return counter


def get_counter() -> Dict:
    return Counter({"all": 0, "failed": 0})


def print_report(counter) -> None:
    print()
    print(f"Imported {counter['all']} samples.")
    if counter["failed"] > 0:
        print(
            f"Skipped {counter['failed']} samples that had transcript longer than the audio."
        )
    else:
        print(Fore.GREEN + "No sample was skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rows with longer output (transcrpt) than input (audio)"
    )
    parser.add_argument("csv_file", help="Path to the csv file with outliers")
    parser.add_argument("--clips_dir", help="Path to the clips folder", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    # Ensure exactly two arguments
    if len(sys.argv) < 3:
        print(
            "Usage: python remove_outliers.py /path/to/input/csv/file /path/to/clips/directory"
        )
        sys.exit(1)

    INPUT_FILE = PARAMS.csv_file
    CLIPS_DIR = PARAMS.clips_dir

    # check if input file exists
    if not os.path.exists(INPUT_FILE):
        print("Error: Cannot find input file. Please specify a valid file path")
        sys.exit(1)

    # check input file type
    if not (INPUT_FILE.endswith(".csv")):
        print("Error: Input file must be a CSV file")
        sys.exit(1)

    # check if the clips directory is a directory
    if not os.path.isdir(CLIPS_DIR):
        print("Error: Specified clips path is not a directory")
        sys.exit(1)

    if not os.path.exists(CLIPS_DIR):
        print(
            "Error: Cannot find clips directory. Please specify a valid directory path"
        )
        sys.exit(1)

    counter = remove_outliers(INPUT_FILE, CLIPS_DIR)
    print_report(counter)
