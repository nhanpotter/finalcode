import csv

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

with open('/Users/michellevanessa/Desktop/automatic-text-scoring-master/Final Code and Data/Augmented_Feat.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        for x in row:
            if '\xc3' in x:
                print(x)
print('done')