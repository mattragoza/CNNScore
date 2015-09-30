SOLVER = ["""\
net: \"""", "<net>", """\"

test_iter: 100
test_interval: 1000

base_lr: 0.01
momentum: 0.9

lr_policy: "fixed"

display: 100
max_iter: 10000
solver_mode: GPU
"""]

if __name__ == "__main__":

	for i in range(10):

		name = "solver/NNScore/nnscore_solver_" + str(i) + ".prototxt"
		net  = "model/NNScore/nnscore_model_" + str(i) + ".prototxt"
		solver = SOLVER[0] + net + SOLVER[2]

		solver_file = open(name, "w")
		solver_file.write(solver)
		solver_file.close()
