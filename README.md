Contents:

* agent.py
    * Agent
* model.mdl
    * Weights for the agent
* train.py
    * Code for training the agent against Simple-AI
* train_finetune.py
    * Code for finetuning the model against itsels, while also playing with the other agents
* plot_data
    * Folder for storing plots when running training
* test_agent_weights
    * Folder to store test agent weights
* test_agents
    * Slightly modified code of the TA provided test agents
* train_weights
    * Weights stored throughout training
* wimblepong
    * The environment so that training can be run out the box

Instructions:

* Training from scratch against Simple-AI can be run with "python train.py" in the root folder
* Finetuning the "model.mdl" against the agent itself, Simple-AI, and the test agents can be run with "python train_finetune.py" in the root folder