import os

def split_csv_by_annotator(csv_path="data/StudEmo_annotation_data.csv"):
    # Create 25 output file handles, one for each annotator_id (0..24)
    out_files = []
    for annot_id in range(25):
        f = open(f"data/annotation_data_annotator_{annot_id}.csv", "w", encoding="utf-8")
        out_files.append(f)
    
    with open(csv_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
    
    # The first line is the header - write it to each annotator file
    header = lines[0]
    for f in out_files:
        f.write(header)
    
    # Process all subsequent lines
    for line in lines[1:]:
        columns = line.split(",", 2)
        if len(columns) < 2:
            continue
        
        # The second column is annotator_id
        try:
            annot_id = int(columns[1])
        except ValueError:
            continue
        
        # Write this entire line to the appropriate file
        out_files[annot_id].write(line)
    
    # Close all output files
    for f in out_files:
        f.close()

if __name__ == "__main__":
    split_csv_by_annotator("data/StudEmo_annotation_data.csv")
