import os

def count_line_in_file(file_path, target_line):
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.rstrip('\n') == target_line:
                    count += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return count

def traverse_and_count(folder, target_line):
    total_count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            total_count += count_line_in_file(file_path, target_line)
    return total_count

def main():
    # Hardcoded variables: change these as needed
    folder = "GNN_Heuristics/data/sokoban"  # Directory to search
    target_line = "END_LABELED_STATE"  # Change this to the line you want to count

    total = traverse_and_count(folder, target_line)
    print(f"Total occurrence of the specified line: {total}")

if __name__ == "__main__":
    main()
