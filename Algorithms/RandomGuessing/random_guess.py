import h5py
import numpy as np

f = h5py.File('../../data/Outfile_CellInformation.h5','r')

n_events = len(f['RealRes_TotalEnergy_Layer1'][:])

geometry = {}

for layer_i in range(6):
    geometry[layer_i+1] = f['RealRes_TotalEnergy_Layer'+str(layer_i+1)][0].shape[1]

outputfile = h5py.File('randomguess_output.h5', 'w')

output_arrays = {}
h5_datasets = {}

for layer_i in range(6):
	output_arrays['Predicted_NeutralEnergy_Layer'+str(layer_i+1)] = []
	h5_datasets[layer_i+1] = outputfile.create_dataset('Predicted_NeutralEnergy_Layer'+str(layer_i+1), (n_events,1,geometry[layer_i+1],geometry[layer_i+1]))


buffer_size = 1000
for event_i in range(n_events):

	input_array = []
	for layer_i in range(6):
		layer_array = f['RealRes_TotalEnergy_Layer'+str(layer_i+1)][event_i]
		input_array.append(layer_array)

	for layer_i in range(6):
		input_i = input_array[layer_i]
		shape_i = input_i.shape
		#pick some random fraction between 0,1
		output_i = np.random.rand(shape_i[0],shape_i[1],shape_i[2])

		#get the output in terms of energy out of the total
		output_i = input_i*output_i

		output_arrays['Predicted_NeutralEnergy_Layer'+str(layer_i+1)].append(output_i)

	if event_i > 0 and event_i % buffer_size == 0 or event_i==n_events-1:
		for layer_i in range(6):
			array_to_save = np.array(output_arrays['Predicted_NeutralEnergy_Layer'+str(layer_i+1)])
			n_buffered = len(array_to_save)
			h5_datasets[layer_i+1][event_i-n_buffered:event_i] = array_to_save
			output_arrays['Predicted_NeutralEnergy_Layer'+str(layer_i+1)] = []

outputfile.close()





