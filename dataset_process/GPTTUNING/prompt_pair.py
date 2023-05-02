import csv
import json

def process_data(input_csv, output_json):
    prompt_completion_pairs = []

    with open(input_csv, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            summary = row["summary"]
            genre = row["genre"]

            prompt = summary + "\n\n###\n\n"
            completion = " " + genre

            pair = {"prompt": prompt, "completion": completion}
            prompt_completion_pairs.append(pair)

    with open(output_json, "w") as jsonfile:
        json.dump(prompt_completion_pairs, jsonfile, ensure_ascii=False, indent=2)

input_csv = "genre_data.csv"
output_json = "cli_ready.json"

process_data(input_csv, output_json)
