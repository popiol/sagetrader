# sagetrader

AWS Sagemaker project for financial analysis.

## Getting the data

The data is pulled from the IB Web API:

https://interactivebrokers.github.io/cpwebapi/

Logging into the API is handled with the help of the IBeam project:

https://github.com/Voyz/ibeam

There are two kinds of data pulled: 
* historical daily bars
* real-time bid/ask/trade data

The data is stored in an S3 bucket and used for training and prediction of two separate models.
One model is based on the historical data and picks the currently most interesting companies.
The second model takes advantage of the real-time data to make the decision to buy or sell shares.

## Reinforcement learning

The project uses RL methods in order to maximize the expected income generated by the model.
The models are built with the Keras library, and the RL environment is built in compliance
with the OpenAI Gym library.

Both the models are mapping an environment state into an action.
For finding the optimal policy we use a simple approach of updating the model 
whenever the reward (immediate or total) is sufficiently high (compared to the average reward).

The state of the environment is defined as the sequences of quotes for each company (daily bars or real-time data).

The action for the first model is defined as the confidence for each company to bring profit.
The action for the second model is defined as:
* the buy price for each company picked by the first model
* the sell price for each company that is currently in the portfolio

The reward function is defined as the income relative to the capital in the previous step.
