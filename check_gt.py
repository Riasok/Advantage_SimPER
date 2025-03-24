import json
from collections import Counter

def check_missing_ground_truth(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check overall structure
    print(f"Total rows in JSON: {len(data)}")
    
    # Sample of keys in first row
    if data:
        first_row = data[0]
        print(f"Sample keys in first row: {list(first_row.keys())}")
    
    # Check for ground truth fields
    possible_ground_truth_keys = ['ground_truth_answer', 'answer', 'gold_answer']
    
    # Count missing ground truth fields
    missing_ground_truth = 0
    available_keys_counter = Counter()
    
    for row in data:
        has_ground_truth = False
        
        # Record all keys that exist in this row
        for key in row.keys():
            available_keys_counter[key] += 1
            
        # Check for ground truth fields
        for key in possible_ground_truth_keys:
            if key in row and row[key]:
                has_ground_truth = True
                break
        
        if not has_ground_truth:
            missing_ground_truth += 1
    
    # Report findings
    print(f"\nRows missing ground truth answers: {missing_ground_truth} out of {len(data)} ({missing_ground_truth/len(data)*100:.2f}%)")
    print("\nFrequency of keys in the dataset:")
    for key, count in available_keys_counter.most_common():
        print(f"  {key}: {count} rows ({count/len(data)*100:.2f}%)")
    
    # If ground truth is consistently missing, suggest alternative approaches
    if missing_ground_truth > 0:
        print("\nSuggestion: Look at a complete sample row to find where ground truth might be stored:")
        if data:
            print(json.dumps(data[0], indent=2)[:1000] + "..." if len(json.dumps(data[0], indent=2)) > 1000 else json.dumps(data[0], indent=2))

if __name__ == "__main__":
    json_file = "llamadpo.json"  # Replace with your actual file path
    check_missing_ground_truth(json_file)