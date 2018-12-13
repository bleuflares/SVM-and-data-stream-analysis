import sys
from pyspark import SparkConf, SparkContext


#mapper for normalizing each column of matrix
def classifier(pair):
    matmul = 0.0
    for point in pair[0]:
        for w_i in w:
            matmul += points * w_i

    delta = []
    if (matmul + b) *pair[1] < 1:
        for j in range(len(w)):
            delta.append((j, (-1) * pair[1] * pair[0][j]))
        delta.append((len(w) + 1, (-1) * pair[1]))
    else:
        for j in range(len(w) + 1):
            delta.append((j, 0.0))
    return delta

def validate(pair):
    matmul = 0.0
    for point in pair[0]:
        for w_i in w:
            matmul += points * w_i
    delta = []
    if (matmul + b) *pair[1] < 1:
        return (1, 1)
    else
        return (0, 1)

def gd(pair):
    if pair[0] < len(w):
        return(pair[0], w[pair[0]] - lr * (w[pair[0]] + reg * pair[1]))
    else:
        return(pair[0], b - lr * (b + reg * pair[1]))

if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    input_file1 = open(sys.argv[1], 'r')
    input_file2 = open(sys.argv[2], 'r')

    feature_points = []
    for line in input_file1:
        point = line.split(',')
        for i in range(len(point)):
            point[i] = int(point[i])
        feature_points.append(point)

    label_points = []
    for line in input_file2:
        label_points.append([int(line)])

    
    lr = 0.1
    reg = 0.1
    w = [0.0 for i in range(122)]
    b = 0.0

    batch_results = []
    for k in range(minbatch):
        fl_pairs = []
        for i in range(6000 / minbatch):
            feature_set = feature_points[6000 / minbatch * k : 6000 / minbatch * (k + 1)]
            label_set = label_points[6000 / minbatch * k : 6000 / minbatch * (k + 1)]
            fl_pairs.append((feature_set[i], label_set[i]))

        features = sc.parallelize(fl_pairs)
        
        deltas = features.flatmap(classifier)
        deltasum = deltas.reduceByKey(lambda a, b: a + b)
        trained = deltasum.map(gd)
        batch_results.append(trained.collect())
    avg_w []
    for i in range(len(w)):
        avg_sum = 0.0
        for j in range(len(minbatch)):
            avg_sum += batch_results[j][i][1]
        avg_w.append(avg_sum/len(minbatch))
    avg_b = 0.0
    for j in range(len(minbatch)):
        avg_b += batch_results[j][len(w) + 1][1]
    avg_b = avg_b / len(minbatch)

    w = avg_w
    b = avg_b
    
    #validation                
    val_pairs = []
    for i in range(len(feature_points))
        fl_pairs.append((feature_points[i], label_points[i]))

    features = sc.parallelize(fl_pairs)
    scores = features.map(validate).reduceByKey(lambda a, b: a + b)
    print(scores.collect())

    sc.setLogLevel('WARN')
    sc.stop()
