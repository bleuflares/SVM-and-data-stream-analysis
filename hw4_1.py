import sys
import numpy as np

if __name__ == "__main__":
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

	
	logs = []
	lrset = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
	regset = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 100]

	lropt = lrset[0]
	regopt = regset[0]
	optscore = 1.0

	for lr in lrset:
		for reg in regset:
			diff_count = 0

			for k in range(10):
				train_features = feature_points[: 600 * k]
				train_features.extend(feature_points[600 * (k + 1) : ])
				train_labels = label_points[: 600 * k]
				train_labels.extend(label_points[600 * (k + 1)  : ])
				train_feature_set = np.array(train_features)
				train_label_set = np.array(train_labels)
				test_feature_set = np.array(feature_points[600 * k : 600 * (k + 1)])
				test_label_set = np.array(label_points[600 * k : 600 * (k + 1)])

				
				row = len(train_features)
				col = len(train_features[0])
				w = np.ones((col, 1))
				b = 0

				print("set %d" %(k))
				while(1):
					matmul = np.dot(train_feature_set, w) + b
					delta = np.zeros((col, 1))
					b_delta = 0.0
					for i in range(row):
						if train_label_set[i][0] == 1:
							if matmul[i][0] < train_label_set[i][0]:
								for j in range(col):
									delta[j] = delta[j] - train_label_set[i][0] * train_feature_set[i][j]
								b_delta = b_delta - train_label_set[i][0] # used to be j but which is correct???
						elif train_label_set[i][0] == -1:
							if matmul[i][0] > train_label_set[i][0]:
								for j in range(col):
									delta[j] = delta[j] - train_label_set[i][0] * train_feature_set[i][j]
								b_delta = b_delta - train_label_set[i][0]

					w_old = w
					b_old = b
					w = w - lr * (w + reg * delta)
					b = b - lr * (b + reg * b_delta)
					count  = 0
					for k in range(col):
						if (w_old[k] - w[k]) < 0.001:
							count += 1
					if count == col:
						break

				try_label = np.dot(test_feature_set, w) + b
				for i in range(600):
					if try_label[i][0] * test_label_set[i][0] < 1:
						diff_count += 1

			score = float(diff_count) / 6000

			logs.append((lr, reg, score))
			print(lr, reg, score)
			if score < optscore:
				optscore = score
				lropt = lr
				regopt = reg
	
	print(lropt, regopt, 1.0 - optscore)
