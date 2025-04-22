import pandas as pd

def filter_first_occurrences(input_file, output_file):
    # Track seen filenames
    seen_files = set()
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Extract the filename (first component before comma)
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue  # Skip malformed lines
                
            filename = parts[0]
            
            # If we haven't seen this filename before, write it to output
            if filename not in seen_files:
                seen_files.add(filename)
                outfile.write(line)
    
    print(f"Processed {len(seen_files)} unique files")

# Example usage
filter_first_occurrences('/home/balamurugan.d/src/test_250311_0x.csv', '/home/balamurugan.d/src/test.csv')