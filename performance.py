import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from models.deepnetwork.core.intermediate import DeepNetwork 
 

models = {
    "VGG-16": "examples/deepnetwork/VGG16.pmml",
    "ResNet-50": "examples/deepnetwork/ResNet50.pmml",
    "MobileNet-224": "examples/deepnetwork/MobileNet.pmml",
    "DenseNet-121": "examples/deepnetwork/DenseNet121.pmml",
}

machine_type = "CPU"
results_file = "tests/performance.json"
image_file = "tests/assets/cat.jpg"

N_EVAL = 1000
TPU_WORKER = "maxkferg"
TPU_CORES = 8

def load_results():
    if not os.path.exists(results_file):
        with open(results_file,'w') as fd:
            json.dump({}, fd)
    with open(results_file,'r') as fd:
        results = json.load(fd)
    # Setup the structure of the datatype
    results.setdefault(machine_type,{})
    return results



def save_results(results):
    with open(results_file,'w') as fd:
        json.dump(results, fd, indent=4) 



def create_load_time_data(model_name, pmml_file):
    """
    Record the amount of time a file load takes
    Data is created for a single type of model
    """
    results = load_results()
    results[machine_type].setdefault(model_name, {})

    # Load PMML file and weights
    start_time = time.time()
    model = DeepNetwork(pmml_file)
    pmml_load_time = time.time()

    # Load weights
    keras_model = model.get_keras_model(load_weights=False)
    weights_start_time = time.time()
    model.load_weights(keras_model)
    complete_time = time.time() 

    # Calculate all the statistics
    results[machine_type][model_name]['total_load_time'] = complete_time - start_time
    results[machine_type][model_name]['pmml_load_time'] = pmml_load_time - start_time
    results[machine_type][model_name]['weight_load_time'] = complete_time - weights_start_time
    # Append the new results to the json file
    save_results(results)



def create_predict_time_data(model_name, pmml_file, image_file, n=1):
    """
    Record the amount of time a prediction takes
    """
    results = load_results()
    results[machine_type].setdefault(model_name, {})
    
    data = imread(image_file)
    model = DeepNetwork(pmml_file)
    model.predict(data, tpu_worker=TPU_WORKER) # Load keras model 
    start_time = time.time()
    # Repeat the predictions multiple times
    for i in range(n):
        result = model.predict(data, tpu_worker=TPU_WORKER)
    print("Model predicted class: %s"%result)
    avg_duration = (time.time() - start_time)/n
    print("Average prediction duration %.2f seconds"%avg_duration)
    results[machine_type][model_name]['predict_time'] = avg_duration
    # Append the new results to the json file
    save_results(results)



def create_all_data(n=1):
    for model_name, pmml_file in models.items():
        create_load_time_data(model_name, pmml_file)
        create_predict_time_data(model_name, pmml_file, image_file, n=n) 



def bar_plot(labels, *columns):
    # set width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = columns[0]
    bars2 = columns[1]
    bars3 = columns[2]
     
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r = [r1,r2,r3]
     
    colors = ['#7f6d5f', '#557f2d', '#2d7f5e']
    names = ["CPU", "GPU", "TPU"]

    # Make the plot
    for i,bar in enumerate(columns):
        plt.bar(r[i], bar, color=colors[i], width=barWidth, edgecolor='white', label=names[i])

    # Add xticks on the middle of the group bars
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], labels)
     
    # Create legend & Show graphic
    plt.legend()



def create_plots():
    machine_types = ["CPU","GPU","TPU"]
    
    total_load_time = [[] for _ in machine_types]
    pmml_load_time = [[] for _ in machine_types]
    weight_load_time = [[] for _ in machine_types]
    predict_time = [[] for _ in machine_types]

    for i,machine_type in enumerate(machine_types):
        results = load_results()[machine_type]
        models = list(results.keys())

        for model in models:
            total_load_time[i].append(results[model]['total_load_time'])
            pmml_load_time[i].append(results[model]['pmml_load_time'])
            weight_load_time[i].append(results[model]['weight_load_time'])
            predict_time[i].append(results[model]['predict_time'])
        
    # Total load time
    plt.figure()
    bar_plot(models, *total_load_time)
    plt.ylabel("Total load time (s)")

    # Total load time
    plt.figure()
    bar_plot(models, *pmml_load_time)
    plt.ylabel("PMML load time (s)")

    # Total load time
    plt.figure()
    bar_plot(models, *weight_load_time)
    plt.ylabel("Weight load time (s)")
    
    # Total load time
    plt.figure()
    bar_plot(models, *predict_time)
    plt.ylabel("Prediction Time (s)")
    plt.show()


    


if __name__ == "__main__":
    create_all_data(n=N_EVAL)
    create_plots()

