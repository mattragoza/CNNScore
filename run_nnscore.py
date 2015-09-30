import os
import datetime

if __name__ == "__main__":

	for i in range(10):

		os.system("caffe train \
			--model=model/NNScore/nnscore_model_" + str(i) + ".prototxt \
			--solver=solver/NNScore/nnscore_solver_" + str(i) + ".prototxt \
			2>&1 | tee logs/nnscore_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +".log")