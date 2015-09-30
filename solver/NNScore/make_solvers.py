SOLVER = ["""\
net: \"""", "model/NNScore/nnscore_model_0.prototxt", """\"

test_iter: 1000
test_interval: 5000

base_lr: 0.01
momentum: 0.9

lr_policy: "fixed"

display: 1000
max_iter: 100000
solver_mode: GPU
"""]

if __name__ == "__main__":

	for i in range(10):

		name = "nnscore_solver_" + str(i) + ".prototxt"
		net  = "model/NNScore/nnscore_model_" + str(i) + ".prototxt"
		solver = SOLVER[0] + net + SOLVER[2]

		solver_file = open(name, "w")
		solver_file.write(solver)
		solver_file.close()
