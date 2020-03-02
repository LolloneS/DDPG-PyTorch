# DDPG-PyTorch
Deep Deterministic Policy Gradient || PyTorch || OpenAI Gym

Lorenzo Soligo, Ca' Foscari University of Venice. Project for the *Artificial Intelligence: Machine Learning and Pattern Recognition* course.

## Explanation
### Transition
`dataclass` that represents a `(s, a) --> (r, s')` transition, and specifies whether `s'` is a terminal state.

### Variables
File that contains all the fundamental parameters from the paper and computes the `device` to be used by PyTorch.


### Noise
Implements a standard Gaussian noise.
