#!/bin/bash

for kinematic_set in {1..5}; do  # Adjust for the number of kinematic settings
    for replica in {1..1000}; do
        python run_replica_method.py $kinematic_set $replica &
    done
done

wait  # Ensure all processes finish before exiting the script
