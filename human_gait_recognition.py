from load_HuGaDB_file import load_HuGaDB_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, nan_euclidean_distances
import pickle

    
def get_data(path, threshold, activity_type, from_row, to_row, from_col = 0, window_width = 5, window0_width = 2, show_plot = False):
    """
    Function that retrieves IMU data from a file.

    :activity_type: 1 or 2 (walking or running)
    :return: dataframe
    """ 
    new_step = False
    x_steps_gyro = []
    y_steps_gyro = []
    z_steps_gyro = []
    x_step_gyro = []
    y_step_gyro = []
    z_step_gyro = []
    x_steps_acc = []
    y_steps_acc = []
    z_steps_acc = []
    x_step_acc = []
    y_step_acc = []
    z_step_acc = []
    zeros = [[0,0,0,0,0,0]] * (window0_width+1)
    zeros = pd.DataFrame(zeros)

    array = load_HuGaDB_file(path)    # load a file
    array = pd.DataFrame(array)     # create dataframe
    array = array.iloc[from_row:to_row,from_col:from_col+6]    # select preffered data
    array.columns = [0,1,2,3,4,5] # rename columns for concatenating
    array = pd.concat([array, zeros], ignore_index=True) # add zeros to the end so the algorithm can retrieve last step
    array_unchanged = array.copy()

    array[0]+=8000
    array[1]+=5000
    array[2]-=14000

    acc_scaling_factor=1670
    gyro_scaling_factor = 939

    array[0] = array[0]/acc_scaling_factor
    array[1] = array[1]/acc_scaling_factor
    array[2] = array[2]/acc_scaling_factor
    array[3] = array[3]/gyro_scaling_factor
    array[4] = array[4]/gyro_scaling_factor
    array[5] = array[5]/gyro_scaling_factor

    for i in range(len(array)-window_width): #iterate through x, y, z axis with and check summed distance from 0 baseline. 
        x = array.iloc[i:i+window_width, 3]
        y = array.iloc[i:i+window_width, 4]
        z = array.iloc[i:i+window_width, 5]
        x = x.abs().sum()
        y = y.abs().sum()
        z = z.abs().sum()
        sum = x + y + z
        if sum < threshold:     # if summed distance is lesser than threshold, set values in window to 0
            array.iloc[i:i+window_width, :] = 0

    for i in range(len(array)-window0_width):      # iterate through data, append data to lists until 2 next values are 0, then append those lists to "mother-lists" and reset the "child-lists", then repeat
        x_gyro = array.iloc[i:i+window0_width, 3].sum()
        if x_gyro!=0:
            x_acc = array.iloc[i, 0]
            y_acc = array.iloc[i, 1]
            z_acc = array.iloc[i, 2]
            x_gyro = array.iloc[i, 3]
            y_gyro = array.iloc[i, 4]
            z_gyro = array.iloc[i, 5]
            x_step_acc.append(x_acc)
            y_step_acc.append(y_acc)
            z_step_acc.append(z_acc)
            x_step_gyro.append(x_gyro)
            y_step_gyro.append(y_gyro)
            z_step_gyro.append(z_gyro)
            new_step = True
        else:
            if new_step:
                x_steps_gyro.append(x_step_gyro.copy())
                y_steps_gyro.append(y_step_gyro.copy())
                z_steps_gyro.append(z_step_gyro.copy())
                x_steps_acc.append(x_step_acc.copy())
                y_steps_acc.append(y_step_acc.copy())
                z_steps_acc.append(z_step_acc.copy())
                x_step_gyro.clear()
                y_step_gyro.clear()
                z_step_gyro.clear()
                x_step_acc.clear()
                y_step_acc.clear()
                z_step_acc.clear()
            new_step = False

    if show_plot:
        fig, (plt1, plt2, plt3) = plt.subplots(3) # plots raw data, processed data and first step
        fig.tight_layout()
        plt1.plot(array_unchanged, label=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'])
        plt1.legend(loc='upper right')
        plt1.set_title("Raw data")
        plt2.plot(array)
        plt2.set_title("Processed data")
        plt3.plot(x_steps_gyro[0])
        plt3.set_title("First step's gyro_x data")
        plt.show()

    x_steps_gyro.pop(0) # deleting first step, because it's usually disturbed
    y_steps_gyro.pop(0)
    z_steps_gyro.pop(0)
    x_steps_gyro.pop(-1) # deleting last step, because it's sometimes disturbed
    y_steps_gyro.pop(-1)
    z_steps_gyro.pop(-1)

    x_steps_acc.pop(0)
    y_steps_acc.pop(0)
    z_steps_acc.pop(0)
    x_steps_acc.pop(-1)
    y_steps_acc.pop(-1)
    z_steps_acc.pop(-1)

    steps_lenghts = list(map(len, x_steps_gyro)) # get lenght of each step (used as one of features for algorithm) (done only for x_gyro because each axies has the same lenght)
    steps_lenghts_mean = np.mean(steps_lenghts) # get mean lenght of step (used for deviation detection)

    x_steps_gyro = list(map(np.mean, x_steps_gyro)) # get mean value of each step
    y_steps_gyro = list(map(np.mean, y_steps_gyro))
    z_steps_gyro = list(map(np.mean, z_steps_gyro))
    x_steps_acc = list(map(np.mean, x_steps_acc))
    y_steps_acc = list(map(np.mean, y_steps_acc))
    z_steps_acc = list(map(np.mean, z_steps_acc))
    
    target = pd.DataFrame(activity_type for i in range(len(x_steps_gyro))) # create target dataframe

    # check for deviation. If step is significantly longer/shorter than mean, delete its data.
    for i in steps_lenghts:
        if i < 0.5 * steps_lenghts_mean or i > 1.5 * steps_lenghts_mean:
            index = steps_lenghts.index(i)
            x_steps_gyro.pop(index)
            y_steps_gyro.pop(index)
            z_steps_gyro.pop(index)
            x_steps_acc.pop(index)
            y_steps_acc.pop(index)
            z_steps_acc.pop(index)
            steps_lenghts.pop(index)
            target.drop(index, inplace=True)
            target = target.reset_index(drop=True)

    return [x_steps_gyro, y_steps_gyro, z_steps_gyro, x_steps_acc, y_steps_acc, z_steps_acc, steps_lenghts, target]

# first person
data1 = get_data(path='HuGaDB_v2_various_01_07.txt', from_row=931, to_row=1762, activity_type=1, threshold=3.5, show_plot=True)
data2 = get_data(path='HuGaDB_v2_various_01_13.txt', from_row=1000, to_row=2008, threshold=3.7, activity_type=2, window_width=2)

# second person
data3 = get_data(path='HuGaDB_v2_various_02_00.txt', from_row=3371, to_row=5054, threshold=1.28, activity_type=1, window_width=2)
data4 = get_data(path='HuGaDB_v2_various_02_00.txt', from_row=5055, to_row=5472, threshold=3.7, activity_type=2, window_width=2)

# third person
data5 = get_data(path='HuGaDB_v2_various_06_26.txt', from_row=1, to_row=646, threshold=0.85, activity_type=1, window_width=2)
data6 = get_data(path='HuGaDB_v2_various_06_26.txt', from_row=647, to_row=4067, threshold=3.7, activity_type=2, window_width=2)

# fourth person
data7 = get_data(path='HuGaDB_v2_various_08_22.txt', from_row=1, to_row=803, threshold=2.66, activity_type=1, window_width=3)
data8 = get_data(path='HuGaDB_v2_various_08_22.txt', from_row=804, to_row=2372, threshold=6.7, activity_type=2, window_width=3)

df1 = pd.DataFrame(data1[:-1]).transpose() # create dataframes without target column
df2 = pd.DataFrame(data2[:-1]).transpose()
df3 = pd.DataFrame(data3[:-1]).transpose()
df4 = pd.DataFrame(data4[:-1]).transpose()
df5 = pd.DataFrame(data5[:-1]).transpose()
df6 = pd.DataFrame(data6[:-1]).transpose()
df7 = pd.DataFrame(data7[:-1]).transpose()
df8 = pd.DataFrame(data8[:-1]).transpose()

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8]) # concatenate dataframes
target = pd.concat([data1[-1], data2[-1], data3[-1], data4[-1], data5[-1], data6[-1], data7[-1], data8[-1]]) # concatenate target dataframes

# reset indexes so the numeration is continous
df = df.reset_index(drop=True)
target = target.reset_index(drop=True)

# set target column name to 'target'
target.rename(columns = {0:'target'}, inplace = True)

df = pd.concat([df, target], axis=1)

# scale data for better performance
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
scaled_data = scaler.transform(df.drop('target', axis=1))
scaled_data = pd.DataFrame(scaled_data)

X = scaled_data
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56) # make a train-test split to train the model properly
knn = KNeighborsClassifier(n_neighbors=15) # create a classifier
model = knn.fit(X_train, y_train) # train the classifier
pred = knn.predict(X_test) # test the classifier
print(confusion_matrix(y_test, pred)) # print out a confusion matrix, classification report and accuracy score to check how did the algorithm do.
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


# with open('modelmean','wb') as f:
#     pickle.dump(model,f)


# ***CODE BELOW IS USED FOR ELBOW METHOD***

# error_rate = []

# for i in range(1,30):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred = knn.predict(X_test)
#     error_rate.append(np.mean(pred != y_test))

# plt.plot(range(1,30), error_rate, marker = 'o')
# plt.title('Error rate vs n_neighbours value')
# plt.xlabel('n_neighbours value')
# plt.ylabel('Error rate')
# plt.show()
