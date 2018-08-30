# All of the (Traffic) Lights – Q-Learning for Urban Traffic Optimization

## Authors

Dan Amir,
Omer Dolev,
May Yaaron,
Netai Benaim 


## Introduction and Usability
This project RL agent for traffic light control using the SUMO traffic simulation engine.

Training and Inference should be performed using the provided Dockerfile-s only.
Full Training is available only with the TraCI Docker image, since no collisions information is supplied via
libsumo. TraCI GUI was tested only on our on machine - running gui inside docker will require some more setup
steps.



#### Use Instructions
 1. Build image from one of the provided dockerfiles.
 2. Run bash in a container with this directory mounted to "/AI_project"
 3. Run "cd /AI_project"
 4. Run the train_model.py script with one of the provided example configurations or you own configuration
 json file.
 3. logs will be automatically saved to the logs directory


## Short explanation about the supplied directory
The following subdirectories are part of the root directory:
- `configurations`: Includes some examples of training and testing configurations
- `docker`: Dockerfiles used for running our code
- `nets`: Definitions of different road networks used for experimentation and some code for automatic
generation of vehicle route plans.
- `logs`: outputs from experiments runs. Each subdirectory includes the configuration file used for the
experiment, data logging and weights of the trained model.

## Short explanations about the supplied code
Those are some of the main python files in the supplied code base
- `train_model.py`: The main script used for running all the trainings and tests
- `solver.py`: Manages the agents training
- `simulation_runner.py`: Implements classes used to communicate between the agent and the simulation using
abstraction of states, rewards and actions.
- `state_representation.py`: Classes for extracting state representation from the simulation
- `reward_extractor.py`: Same but for rewards
- `sumo_backend.py`: our own unified API for TraCi and libsumo
Different agent classes are implemented in separate files with "agent" in their names.

A bash script named run_exps.sh is provided with some example commands.

## Explanation about configuration parameters

- "backend": One of "traci", "libsumo" and "traci-gui" the interface used for running the simulation.
- "observation_radius": The distance from the junction from which the observation is gathered in meters
- "sumo_cmd": The command used to initiate SUMO
- "net_dir": Path to the directory of the routes network
- "flow_probabilities": dictionary of per step generation probability for each car type in each possible route. In our final results we used only one type.
- "default_program": Default phases duration program for the agent controlled junctions. Relevant only for
- automatic.
- "epsilon": Epsilon greedy parameter
- "actions_type": One of "phase", "binary" and "automatic" as explained in the report.
- "alpha": Learning rate
- "discount": Discount gamma factor
- "r_weight": As explained in the report
- "junctions": List of junction IDs for junctions where reward components will be monitored
- "junctions_with_agents": List of junctions with agents performing actions.
- "num_episodes":  Number of episodes to run
- "steps_per_episode": Number of steps per episode
- "log_path": Path to save logs into.
- "fixed_action": Fixed action number for the agent. Relevant only if mode is "fixed"
- "log": Whether to save logs or not.
- "exploring": If True, uses epsilon greedy decay for the agent.
- "mode": one of "linear", "lqf" and "fixed"
- "reward_weights": Dictionary with weight for each reward component ("wait", "flicker", "emergency", "delay",
 "teleport").
- "simulation_offset": Number of steps to run with automatic agent before every episode.
- "min_phase_duration": Limits the minimal duration before phase change
- "alpha_decay": (optional) Decay factor to multiply by alpha every epoch.
- "load_weights" (optional) List of paths to the weight numpy arrays of the agents (loaded before training).


