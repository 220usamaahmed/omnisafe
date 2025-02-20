# Extending Omnisafe

## Adding new Environments

Omnisafe has a tutorial on adding new environments present in `tutorials/English/3.Environment Customization from Scratch.ipynb` and `./tutorials/English/4.Environment Customization from Community.ipynb`. You can follow these to create new environment from scratch or create a wrapper for an existing environment (from OpenAI Gym for example).

I have added the `Pendulum-v1` environment in `./omnisafe/envs/classic_control/envs_from_gym.py`. Creating environment wrappers for Omnisafe will usually require us to add a cost value along with the existing reward present in RL environments. For this example the cost is zero in all cases.

In case we create a new file we also have to import it in the `./omnisafe/envs/classic_control/__init__.py` file for it to be automatically registed with Omnisafe.

## Adding new Algorithms

Adding new algorithms will usually involve two things: 

1. The actual algorithm which has the specific training logic (in `./omnisafe/algorithms`).
2. The deep learning models the algorithm depends on (in `./omnisafe/models`).

I have added the Deep Q-Learning algorithm to Omnisafe as a simple example.

### Algorithm (`./omnisafe/algorithms/off_policy/dqn.py`)

The algorithm class has to inherit from the `BaseAlgo` class or one of the other subclasses if it is an extention of another algorithm. Inside this class you will have access to the `self._cfgs` variable which will have a dictionary based on the contents of a YAML file of the same name as your class name (`./omnisafe/configs/off-policy/DQN.yaml` in this case). Add all your configurations related to training, model architecture, logger etc. here.

The following methods have to be implemented:

1. `_init_env`: Create an instance of your environment here which will be used during training. Omnisafe provides adapters to wrap your environment. For example, the `OffPolicyAdapter` helps by storing the current observation and allowing us to update the policy before proceeding with the episode. I've haven't used any such adapter to keep things as simple as possible.
2. `_init_model`: Create the deep learning models required for your algorithm. These need to be of type `Actor`. For this example I created the `DiscretizerMLPActor` which is a simple wrapper around a MLP but instead of the output layer matching the size of the action space we have multiple bins per action component. This is because Q-Learning works with discrete actions but we want our algorithm to work on contineous `Box` action spaces. Omnisafe has other model typical models which can be used.
3. `_init`: Initialize the training of your algorithm. Here we initialize our optimizer and a queue to hold the rollback of our environment.
4. `_init_log`: Initialize the logger. This will log different metrics during training. I have ignored this for now. The logger also stores your model weights after training.
5. `learn`: This has the main training loop of your algorithm. Omnisafe provides abstractions for writing the training loop. For example there is a `.rollout` method for collecting episodes. I have choosen to not use any such method to keep things as simple as possible.

We also need to add the algorithm to the `.omnisafe/algorithms/__init__.py` in order for it be registed with Omnisafe.

### Model (`omnisafe/models/actor/discretizer_mlp_actor.py`)

The model class has to inherit from the `Actor` class or one of the other subclasses. For this example I created the `DiscretizerMLPActor`. The only difference between this and `MLPActor` is that the output shape is based on the number of bins specified in order to apply Q-Learning to contineous problems.

The important function that needs to be implemented here is the `predict` method. This will take the observation tensor and output the action tensor. 

During evaluation Omnisafe will create an object of this class and load the model weights. Any PyTorch module (Conv layers, Linear layers etc.) are stored and loaded automatically.

The `forward` and `log_prob` also need to be implemented but they can simply raise `NotImplementedError`.

If we create a new model type we also need to add it the `./omnisafe/models/actor/actor_builder.py` factory. I added an aditional kwargs argument to this factory method to pass the number of bins to use with our model. This method is called from `./omnisafe/evaluator.py`.

### Testing algorithm and environment

The testing code is present at `./examples/use_custom_algo.py`. First run the `train` method and then run the `evaluate` method. Training creates a `runs` folder with the logs and model weights.

```bash

cd omnisafe
pip install -e .

cd examples
python use_custom_algo.py

```