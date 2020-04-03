# DDPG-PyTorch
Deep Deterministic Policy Gradient || PyTorch || OpenAI Gym

Lorenzo Soligo, Ca' Foscari University of Venice. Project for the *Artificial Intelligence: Machine Learning and Pattern Recognition* course.

## Instructions
### Setup
* Create the Conda environment: `conda env create -f environment.yml`
* Activate the environment: `conda activate deeprl`
* Install the requirements from pip: `pip install -r requirements.txt`

### Running the code
`python main.py`

You can use the following flags:
* `--eval`: will run an episode using an already saved model of the actor. Don't use this if you want to train the model.
* `--env`: name of the OpenAI Gym environment to use. The default is `LunarLanderContinuous-v2`. Notice that DDPG is developed to be used with continuous action spaces.

### Running a sample with LunarLander
* Copy the `models` folder from  `results/lunarlander` into the root of the project.
* Run `python main.py --eval` to test LunarLander, or `python main.py --eval --env "AnotherEnv"` to test another environment
    * beware that only `LunarLander` is provided.

## Further information
* This implementation does not precisely follow the one presented in the paper. As a matter of fact, I noticed that not using batch normalization and adding the actions in the critic's input layer drastically improved performance. 
* The `results` folder contains videos, Tensorboard logs and working models for LunarLander

