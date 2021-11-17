# pytorch-chatbot

My first delve into machine learning is this PyTorch chatbot. This project is adapted from the [official PyTorch tutorial]https://pytorch.org/tutorials/beginner/chatbot_tutorial.html and the bot is trained from the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)


### Development
* To start a development environment:
    1. Clone the repository at: https://github.com/CamChandler98/pytorch-chatbot
    2. run the command pipenv install to install dependencies and virtual environment
    
### Chatting
* To chat:
    1. Run the command `pipenv run python chat.py`
    2. Type message into terminal and press enter
    3. type 'q' or 'quit' to end chat function
    
### Reflections and next steps
---
I used a teacher forcing ratio of 50% with 5000 iterations for this bot. I think that these factors combined led to a model that makes similar inferences in disimilar situations. For a future bot I may try mor training iterations with less teacher forcing. Over all I'm prod of being able to implement this bot using just the PyTorch framework but for my next venture into chatbots I think it'd be more effective to incorporate some sort of Natural Language Processing module.
***
