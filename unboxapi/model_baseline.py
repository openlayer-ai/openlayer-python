import csv
import fasttext


def build_model(csv_file: str, text_col: str, label_col: str):
    text_list, label_list = process_csv(csv_file, text_col, label_col)

    label_names = list(set(label_list))
    k = len(label_names)
    label_indices = [i for i in range(k)]
    label_dict = {}

    for i, lbl in enumerate(label_names):
        label_dict[lbl] = i

    label_list_indices = [label_dict[lbl] for lbl in label_list]
    output_file = save_fasttext_file(text_list, label_list_indices)
    return train_model(output_file), k, label_names, label_indices


def process_csv(csv_file: str, text_col: str, label_col: str):
    csv_rows = csv.DictReader(open(csv_file))
    text_list = []
    label_list = []

    for row in csv_rows:
        text = row[text_col]
        label = row[label_col]
        text_list.append(text)
        label_list.append(label)

    return text_list, label_list


def save_fasttext_file(text_list, label_list, output_file="train.data"):
    row_lines = []
    for text, label in zip(text_list, label_list):
        row_line = " ".join(["__label__{0}".format(label), text])
        row_lines.append(row_line)

    with open(output_file, 'w') as f:
        for item in row_lines:
            f.write("%s\n" % item)

    return output_file


def train_model(data_file: str):
    model = fasttext.train_supervised(input=data_file)
    return model
