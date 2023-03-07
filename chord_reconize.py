import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def getPairs(points):
    paris = []
    for ind in range(0, 21):
        paris.append([points[ind*2], points[ind*2+1]])
    return paris, points[-1]


def plotGesture(paris, chord = None):
    plt.scatter(
        [point[0] for point in paris],
        [point[1] for point in paris],
    )

    for i, point in enumerate(paris):
        plt.annotate(i, (point[0], point[1]))
    
    if chord:
        plt.title(chord)

def centralize(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return [[x - x_mean, y - y_mean] for x, y in points]

def rotate(points):
    start_vec = points[0]
    end_vec = points[13]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    angle = np.arctan2(vec[1], vec[0])
    return [[x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle)] for x, y in points]

def vec_length(x, y):
    return np.sqrt(x**2 + y**2)

def scale(points, scl = 0.01):
    start_vec = points[0]
    end_vec = points[13]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    length = np.sqrt(vec[0] ** 2 + vec[1] ** 2) * scl
    return [
        [
            x * np.power(vec_length(x, y), 6) / length,
            y * np.power(vec_length(x, y), 6) / length
        ]
            for x, y in points
    ]

def decompose(points):
    flatten = []
    for point in points:
        flatten.append(point[0])
        flatten.append(point[1])
    return flatten

def process_line(points, scl = 0.01):
    return decompose(scale(rotate(centralize(points)), scl))


def getchord(chord, dataset):
    return dataset.columns[42:][chord == 1].values[0]

class Logger:
    def __init__(self, level=0, class_name='Logger'):
        self.level = level
        self.class_name = class_name
    def log(self, msg, level=0):
        if level >= self.level:
            print(f'[{self.class_name}:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}')
    



class ChordRecognizer:
    def __init__(self, data_path, chords = [str], sal=0.01,  sep=' ', estimators=100, max_depth=6):
        self.logger = Logger(class_name='ChordRecognizer')
        index = []
        for i in range(21):
            index.append(f'{i}x')
            index.append(f'{i}y')
        index.append('chord')
        chord_dataset = pd.DataFrame(columns=index) 
        for (i, chord) in enumerate(chords):
            dataset = pd.read_csv(f'{data_path}/{chord}.csv', sep, header=None)
            dataset.loc[:, 42] = chord
            dataset.columns = index
            chord_dataset = pd.concat([chord_dataset, dataset], axis=0)
        for i, line in enumerate(chord_dataset.values):
            chord_dataset.iloc[i, :42] = process_line(getPairs(line)[0])
        
        self.dataset = pd.get_dummies(chord_dataset, columns=['chord'])
        self.estimators = estimators
        self.max_depth = max_depth
        
    def train(self):
        self.logger.log('start training')
        X = self.dataset.iloc[:, :42].values
        y = self.dataset.iloc[:, 42:].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.max_depth, random_state=0)
        self.clf.fit(X_train, y_train)
        self.logger.log(f'final score {self.clf.score(X_test, y_test)}')

    def predict(self, points):
        return getchord(
            self.clf.predict(
              np.array(process_line(getPairs(points)[0])).reshape(1, -1)
            )[0],
            self.dataset
        )




if __name__ == '__main__':
    # 使用方法
    # 1. 指定文件夹位置
    # 2. 指定和弦名称
    # 3. 指定放缩值
    # 4. 调用train
    # 5. 如果检测到手势之后调用 predict, 向其中注入数组 [1, 2, 3 ,4 ,5, 6, 7,]
    recognizer = ChordRecognizer('./data', ['C', 'E', 'F', 'G'], sal=0.01, max_depth=10, estimators=100)
    recognizer.train()
    print(
        recognizer.predict(
            # 这是一个C
            # [410,279,448,220,491,186,525,163,563,138,498,247,525,267,559,279,590,292,474,293,467,301,454,270,452,246,439,325,423,332,417,299,419,276,403,345,388,356,385,333,388,312]
            # 这是一个E
            # [382,313,408,283,435,265,455,246,470,226,454,306,467,316,471,314,469,311,445,324,440,309,428,294,423,292,427,335,420,316,411,301,407,298,406,342,402,325,397,311,395,306]
            # # 这是一个F
            [437,385,477,360,495,324,490,290,478,267,502,333,496,276,487,242,481,218,473,346,455,276,447,257,446,259,440,353,423,284,423,262,428,261,411,353,402,293,407,277,413,276]
            # # 这是一个G
            # [278,195,283,171,280,144,267,127,251,119,286,171,265,145,258,136,256,132,271,189,242,158,233,142,232,133,257,201,233,171,236,163,242,163,245,208,226,189,230,182,236,180]
        )
    )