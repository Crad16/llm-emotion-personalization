import os

def split_first_50(
    in_dir="data/annotator_split",
    out_prev_dir="data/split_prev",
    out_test_dir="data/split_test"
):
    """
    Reads each CSV file named 'annotation_data_annotator_i.csv' (i=0..24)
    in 'in_dir', then splits the file into two parts:
      - 'annotator_i_prev.csv' -> first 50 lines + header
      - 'annotator_i_test.csv' -> the rest + header
    and saves them to 'out_prev_dir' and 'out_test_dir', respectively.
    """
    os.makedirs(out_prev_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    for annot_id in range(25):
        in_file = os.path.join(in_dir, f"annotation_data_annotator_{annot_id}.csv")
        if not os.path.exists(in_file):
            print(f"File not found: {in_file}")
            continue

        with open(in_file, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        if not lines:
            print(f"No content in file: {in_file}")
            continue

        header = lines[0]
        data_lines = lines[1:]

        prev_part = data_lines[:50]
        test_part = data_lines[50:]

        out_prev = os.path.join(out_prev_dir, f"annotator_{annot_id}_prev.csv")
        out_test = os.path.join(out_test_dir, f"annotator_{annot_id}_test.csv")

        with open(out_prev, "w", encoding="utf-8") as f_prev:
            f_prev.write(header)
            for line in prev_part:
                f_prev.write(line)

        with open(out_test, "w", encoding="utf-8") as f_test:
            f_test.write(header)
            for line in test_part:
                f_test.write(line)

        print(f"Split {in_file} -> {out_prev} / {out_test}")

if __name__ == "__main__":
    split_first_50()
