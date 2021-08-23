# import Libraries.
import numpy as np;
import pandas as pd;
import copy

# --------------------------
# Class Node To Create Tree
class Tree:
    def __init__(self):
        self.name = None
        self.cols = []
        self.rows = []
        self.res = None
        self.children = None
        self.leaf = []


# ----------------------------------------------------------------------------------------------------------------
# Function To Calculate Entropy.
def calculate_entropy(elements):
    entropy = 0;
    for i in range(len(elements)):
        entropy += -((elements[i] / sum(elements)) * np.log2(elements[i] / sum(elements)));
    return entropy;


# ------------------------------------------------------------------------------------------------------------------
# Function To Calculate Information Gain.
def calculate_gain_information(main_entropy, list_elements_entropy, size_elements):
    gain_info = main_entropy;
    for k in range(len(list_elements_entropy)):
        gain_info -= list_elements_entropy[k] * (size_elements[k] / sum(size_elements));
    return gain_info;


# ------------------------------------------------------------------------------------------------------------------
# Open Data_Set and Classified Data To Rows and Clos and Data
def classified_data():

    data = [];  # To Store Data Read From Text File(2D).

    # -----------------------------------------------------------------#
    # Read Data From File Using Pandas (Whole Data in 2D Array).
    dataset = pd.read_csv('house-votes-84.data.txt', header=None);
    dataset = dataset.sample(frac=1)

    r = dataset.values.tolist()
    c = []
    for i in range(len(r[0])):
        tmp = [];
        for j in range(len(r)):
            tmp.append(r[j][i]);
        c.append(tmp);
    data = dataset.iloc[0: len(r), 0:len(c)];
    refineData(r);
    r = editMissingParticipantsInRows(r);
    # -----------------------------------------------#

# Edit Data 2D Array To Replace Missing Participants.
    editMissingParticipantsInData(r);
    # Return 3 Arrays.
    return data, r, c;


def splitData():
    rowsTraining = [];
    columnsTraining = [];
    rowsTesting = [];
    columnsTesting = [];

    # Get Data Refined and Ready For Spliting.
    data, rows, columns = classified_data();
    dataSize = len(rows);

    # Get Training Sets Size Which is 25% of Original Data.
    dataTrainSize = round(dataSize * (25 / 100));

    # Split Data Into Training and Testing Data.
    dataTrain = data.iloc[0: dataTrainSize, 0: len(columns)];
    dataTest = data.iloc[dataTrainSize: len(rows), 0: len(columns)];

    # Split Data Into Training Rows and Columns.
    for i in range(dataTrainSize):
        rowsTraining.append(rows[i]);

    for i in range(len(rowsTraining[0])):
        tmp = [];
        for j in range(len(rowsTraining)):
            tmp.append(rowsTraining[j][i]);
        columnsTraining.append(tmp);

    # Split Data Into Testing Rows and Columns.
    for i in range(dataTrainSize, dataSize):
        rowsTesting.append(rows[i]);

    for i in range(len(rowsTesting[0])):
        tmp = [];
        for j in range(len(rowsTesting)):
            tmp.append(rowsTesting[j][i]);
        columnsTesting.append(tmp);
    # ---------------------------------------------------------------------------------------------------
    return data, rows, columns, dataTrain, dataTest, dataTrainSize, rowsTraining, columnsTraining, rowsTesting, columnsTesting;


def refineData(rows):
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if (rows[i][j] == 'y\n' or rows[i][j] == 'n\n' or rows[i][j] == '?\n'):
                temp = rows[i][j].replace('\n', '');
                rows[i][j] = temp;
    return rows;
# ------------------------------------------------------------------------------------------------------------------
# Function To Set Missing Participants Votes
def editMissingParticipantsInRows(rows):
    result = [];
    for i in range(len(rows)):
        yesCounter = 0;
        noCounter = 0;
        for j in range(len(rows[i])):
            if (rows[i][j] == 'y'):
                yesCounter += 1;
            elif (rows[i][j] == 'n'):
                noCounter += 1;
            else:
                continue;
        flag = 'y';
        if (noCounter > yesCounter):
            flag = 'n';
        for j in range(len(rows[i])):
            if (rows[i][j] == '?'):
                result.append(flag);
                rows[i][j] = flag;

    return rows;
# ------------------------------------------------------------------------------------------------------------------
# Function To Set Missing Participants Votes in Data File.
def editMissingParticipantsInData(rows):
    with open('house-votes-84.data.txt', 'w') as f:
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if (j == len(rows[i]) - 1):
                    f.write("%s" % rows[i][j]);
                else:
                    f.write("%s," % rows[i][j]);
            f.write("\n");

# ------------------------------------------------------------------------------------------------------------------
# Classified Cols To Know Unique Elements And it's Size
def unique_items_size(col):
    unique_item = []
    size_item = []
    for i in range(len(col)):
        key = 0
        for k in range(len(unique_item)):
            if unique_item[k] == col[i]:
                size_item[k] += 1
                key = 1
                break
        if key == 0:
            unique_item.append(col[i])
            size_item.append(1)
    return unique_item, size_item


# ------------------------------------------------------------------------------------------------------------------
# Analyze Data_set (Output for (Yes / No) Counting for Each Feature).
def know_result(col1, col2, element):
    output_items = []
    output_sizes = []
    for i in range(len(col1)):
        key = 0
        if col2[i] == element:
            for j in range(len(output_items)):
                if col1[i] == output_items[j]:
                    key = 1
                    output_sizes[j] += 1
                    break
            if key == 0:
                output_items.append(col1[i])
                output_sizes.append(1)
    return output_items, output_sizes


# ------------------------------------------------------------------------------------------------------------------
# For Each Column -> Get Gain Information Then Choose The Best GI To Be Parent For The Tree.
def max_gain(col):
    output_items, output_sizes = unique_items_size(col[0])
    main_entropy = calculate_entropy(output_sizes)
    total_gain_information = []

    for i in range(1, len(col)):
        unique_items, item_sizes = unique_items_size(col[i])
        entropic = []
        sizes = []
        for j in range(len(unique_items)):
            res_item, size_output = know_result(col[0], col[i], unique_items[j])
            entropic.append(calculate_entropy(size_output))
            sizes.append(sum(size_output))
        total_gain_information.append(calculate_gain_information(main_entropy, entropic, sizes))

    greatest_one = max(total_gain_information)
    place = total_gain_information.index(greatest_one)
    return place + 1,total_gain_information,sorted(total_gain_information)


# ------------------------------------------------------------------------------------------------------------------

def returns_cols(cols, element, place_col):
    main_col = cols[place_col]
    col = []
    for i in range(len(main_col)):
        if element == main_col[i]:
            col.append(i)
    return col


def t (cols):
    max , s, ss = max_gain(cols)
    ss.reverse()
    l = []
    for i in range(len(s)):
        l.append(s.index(ss[i]))
    return l



# Create Decision Tree
def buildTree(root,index):

    if len(index) == 0:
        return root

    place,s,ss = max_gain(root.cols)
    max = root.cols[place]
    root.children, size = unique_items_size(max)
    root.name = index[0]
    index.remove(index[0])
    root.items, root.items_size = unique_items_size(root.cols[0])

    if len(root.items) == 1:
        root.res = root.items[0]
        #print(root.items_size , " " , root.res, " ")
        return
    else:
        root.res = '-1'

        for item in root.children:
            node = Tree()
            c = returns_cols(root.cols, item, place)


            for i in range(len(c)):
                r = copy.deepcopy(root.rows[c[i]])
                r.remove(item)
                node.rows.append(r)

            for i in range(len(node.rows[0])):
                temp = []
                for j in range(len(node.rows)):
                    temp.append(node.rows[j][i])
                node.cols.append(temp)
            root.leaf.append(node)

        for node in root.leaf:
            if len(node.cols) > 1:
                 buildTree(node,index)

def prdict(root,row):
    if(root.name == None):
        return
    res = root.res

    index = int(root.name)

    key = row[index]

    if res != '-1':
         return res
    else:
        get = 0
        for i in range(len(root.children)):
            if root.children[i] == key:
               get = i
               break
        return prdict(root.leaf[get],row)

# ------------------------------------------------------------------------------------------------------------------
# Main Function
def main():


    for j in range(5):
        data, rows, columns, dataTrain, dataTest, dataTrainSize, rowsTraining, columnsTraining, rowsTesting, columnsTesting = splitData();
        node = Tree()
        node.cols = columnsTraining
        node.rows = rowsTraining


        buildTree(node,t(columnsTraining))
        accuary = 0
        for i in range(len(rowsTraining)):
            r = copy.deepcopy(rowsTraining[i])
            answer = copy.deepcopy(r[0])
            r.remove(r[0])
            ans = prdict(node, r)

            if answer == ans:
               accuary += 1

        print("Accuarcy sample ",j+1,accuary / len(rowsTraining) * 100)


if __name__ == "__main__":
    main();