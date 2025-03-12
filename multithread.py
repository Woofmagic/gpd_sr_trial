import multiprocessing as mp
import tensorflow as tf

# Function to limit GPU memory per process
def set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Wrapper function to execute replica training
def train_replica(kinematic_set_number, replica_index, model_builder, data_file):
    set_gpu_memory_growth()
    run_local_fit_replica_method(
        number_of_replicas=1,  # Each process handles one replica
        model_builder=model_builder,
        data_file=data_file,
        kinematic_set_number=kinematic_set_number,
    )

def parallel_run(kinematic_sets, num_replicas_per_set, model_builder, data_file):
    num_processes = min(mp.cpu_count(), len(kinematic_sets) * num_replicas_per_set)

    # Create a list of tasks (each task is a tuple of (kinematic_set_number, replica_index))
    tasks = [
        (kinematic_set, replica_idx, model_builder, data_file)
        for kinematic_set in kinematic_sets
        for replica_idx in range(num_replicas_per_set)
    ]

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(train_replica, tasks)

# Run parallelized replica training
