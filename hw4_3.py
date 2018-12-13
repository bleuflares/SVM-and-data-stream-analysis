import sys
import numpy as np

if __name__ == "__main__":
	input_file1 = open(sys.argv[1], 'r')
	one_pos = []
	timeline = 1
	for line in input_file1:
		if int(line) == 1:
			one_pos.append(timeline)
		timeline += 1

	buckets = []
	bucket_size = 1
	bucket_count = 0
	cur_pos = 0
	while(cur_pos < len(one_pos)):
		if bucket_count == 2:
			bucket_size = bucket_size * 2
			bucket_count = 0

		buckets.append((bucket_size, one_pos[cur_pos]))
		cur_pos += bucket_size
		bucket_count += 1
	
	for k in range(len(sys.argv) - 2):
		estimate = 0
		for i in range(len(buckets)):
			if buckets[i][1] <= int(sys.argv[k + 2]):
				estimate += buckets[i][0]
				i += 1
				continue
			else:
				estimate -= buckets[i - 1][0] / 2
				break
		print(estimate)