import matplotlib.pyplot as plt
from collections import Counter
import os
import csv

labels_dict = {}

def parse_file(file_path):
    global max_label
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "BEGIN_LABELED_STATE" and i + 1 < len(lines):
            label_line = lines[i + 1].strip()
            try:
                label = int(label_line)
                # print(f"Label found: {label}")
                if label not in labels_dict:
                    labels_dict[label] = 0
                labels_dict[label] += 1
            except ValueError:
                print(f"Warning: Unable to parse '{label_line}' as an integer.")
            i += 1
        i += 1

def plot_histogram():
    sorted_labels = sorted(labels_dict.keys())
    max_label = max(sorted_labels)
    print(f"Max label found: {max_label}")
    frequencies = [labels_dict[label] for label in sorted_labels]
    plt.figure(figsize=(10, 6))

    plt.bar(sorted_labels, frequencies, align='center', color='red')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of Label Distribution')
    
    os.makedirs('results_analysis/SOKOBAN', exist_ok=True)
    with open('results_analysis/SOKOBAN/label_distribution.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Label', 'Frequency'])
        for label in sorted_labels:
            writer.writerow([label, labels_dict[label]])

    total_values = sum(frequencies)
    print(f"Sum of frequencies: {total_values}")
    top_labels = sorted(labels_dict.items(), key=lambda item: item[1], reverse=True)[:10]
    print("Top 10 labels by count:")
    for label, count in top_labels:
        print(f"Label {label}: {count}")

    plt.savefig('results_analysis/SOKOBAN/label_distribution_histogram.png')

def main():
    directory_path = "data/sokoban"
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):
            full_file_path = os.path.join(directory_path, file)
            parse_file(full_file_path)
    if labels_dict:
        # print("Labels found in the files:")
        # for label, count in labels_dict.items():
        #     print(f"Label {label}: {count} occurrences")
        plot_histogram()
    else:
        print("No labels found in the files.")

if __name__ == '__main__':
    main()
