import matplotlib.pyplot as plt
import sys
import numpy as np

allcolors = ['b', 'g', 'c', 'r', 'k', 'y']

def get_acc(fin):
    train, val = [], []
    with open(fin) as f:
        for line in f:
            line = line.strip()
            if 'accuracy' in line and 'Validation' not in line:
                tmp = float(line[line.rfind('=')+1:])
            if 'Validation-accuracy' in line:
                val.append(float(line[line.rfind('=')+1:]))
                train.append(tmp)
            if 'Start' in line:
                train, val = [], []
    
    return train[2:], val[2:]

def plot(trains, vals, titles):
    for (train, val, title) in zip(trains, vals, titles):
        print title, val[-1], np.max(val), np.argmax(val), len(val)
    plt.clf()
    colors = allcolors[:len(trains)]
    for (train, val, title, c) in zip(trains, vals, titles, colors):
        x = range(len(train))
        plt.plot(x, train, c, label='%s_train'%title)
        plt.plot(x, train, '%so'%c)
        x = range(len(val))
        plt.plot(x, val, '%s--'%c, label='%s_val'%title)
        plt.plot(x, val, '%so'%c)
        #plt.plot(r, p, label='%s'%(title), linewidth=2, color=c)
    #    print title, val[-1], np.max(val), np.argmax(val), len(val)
    # x = range(len(train))
    # plt.plot(x, train, 'r')
    # plt.plot(x, train, 'ro')
    # x = range(len(val))
    # plt.plot(x, val, 'b')
    # plt.plot(x, val, 'bo')
    #print x, val
    plt.legend(loc="lower right", fontsize=10)
    plt.show()

if __name__ == '__main__':
    mode = sys.argv[2]
    folder = sys.argv[1]
    titles = ['%s%s.log' % (mode, x) for x in sys.argv[3:]]#['cifar100', 'cifar100_factor']
    trains, vals = [], []
    for title in titles:
        train, val = get_acc('%s/log/%s' % (folder, title))
        trains.append(train)
        vals.append(val)

    plot(trains, vals, titles)
                
