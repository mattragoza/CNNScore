if __name__ == "__main__":

	for i in range(10):

		solver_name = "solver/NNScore/nnscore_solver_" + str(i) + ".prototxt"
		model_proto = "model/NNScore/nnscore_model_" + str(i) + ".prototxt"

		solver = """\
net: \""""+model_proto+"""\"

test_iter: 100
test_interval: 1000

base_lr: 0.01
momentum: 0.9

lr_policy: "fixed"

display: 100
max_iter: 10000
solver_mode: GPU
"""
		solver_file = open(solver_name, "w")
		solver_file.write(solver)
		solver_file.close()
