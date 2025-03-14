
import argparse
import os
import csv
import numpy as np

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='This is an example of argument parsing in Python.')

    # Add arguments to the parser
    parser.add_argument('--folder', type=str, help='Your name', required=True)
    parser.add_argument('--method', type=str, help='Your age', required=True)
    parser.add_argument('--model', type=str, help='Your age', required=True)

    # Parse the arguments
    args = parser.parse_args()
    print(args.folder)
    # iterate through folder in folder
    folder_list = os.listdir(args.folder)
    results = []
    num_queries = []
    for folder in folder_list:
        # skip if not a folder
        if not os.path.isdir(f'{args.folder}{folder}'):
            continue
        result_fp = f'{args.folder}{folder}/eval_{args.method}_{args.model}.csv'
        # open csv
        with open(result_fp, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == 'seen_recall':
                    results.append(float(row[1][1:]))
                if row[0] == 'seen_queries':
                    num_queries.append(int(row[1]))
    weighted_results = [results[i]*num_queries[i]/sum(num_queries) for i in range(len(results))]
    average = np.mean(results)
    weighted_average = sum(weighted_results)
    print(average)
    print(weighted_average)

    # create new csv file
    fp = f'{args.folder}average_results_{args.method}_{args.model}.csv'
    print(fp)
    with open(fp, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['average_recall', average])
        writer.writerow(['weighted_average_recall', weighted_average])

if __name__ == '__main__':
    main()