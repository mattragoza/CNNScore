MODEL = ["""\
name: \"""", "<name>", """\"

layer {
  name: "data_layer"
  type: "Data"
  top: "data_blob"
  top: "label_blob"
  data_param {
    source: \"""", "<train>", """\"
    batch_size: 20000
    backend: LMDB
    prefetch: 8
  }
  include: { phase: TRAIN }
}

layer {
  name: "data_layer"
  type: "Data"
  top: "data_blob"
  top: "label_blob"
  data_param {
    source: \"""", "<test>", """\"
    batch_size: 2000
    backend: LMDB
    prefetch: 8
  }
  include: { phase: TEST }
}

layer {
  name: "hidden_sum_layer"
  type: "InnerProduct"
  bottom: "data_blob"
  top: "hidden_sum_blob"

  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "hidden_act_layer"
  type: "Sigmoid"
  bottom: "hidden_sum_blob"
  top: "hidden_act_blob"
}

layer {
  name: "output_sum_layer"
  type: "InnerProduct"
  bottom: "hidden_act_blob"
  top: "output_sum_blob"

  inner_product_param {
    num_output: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "output_act_layer"
  type: "Sigmoid"
  bottom: "output_sum_blob"
  top: "output_act_blob"
}

layer {
  name: "error_layer"
  type: "EuclideanLoss"
  bottom: "output_act_blob"
  bottom: "label_blob"
  top: "error_blob"
}"""]

if __name__ == "__main__":

	for i in range(10):

		name = "model/NNScore/nnscore_model_" + str(i) + ".prototxt"
		train = "lmdb/SCOREDATA.vina.balanced." + str(i) + ".train"
		test = "lmdb/SCOREDATA.vina.balanced." + str(i) + ".test"
		model = MODEL[0] + name + MODEL[2] + train + MODEL[4] + test + MODEL[6]

		model_file = open(name, "w")
		model_file.write(model)
		model_file.close()
