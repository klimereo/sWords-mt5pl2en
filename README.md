# sWords-mt5pl2en
Automatic data annotator powered by mt5-pl2en translation model. The fine-tuned models try to to translate Polish key-meanings into English while also extracting keywords.

## Scripts
`fine_tuner.py` demonstrates the training script of the models. `predictor.py`, on the other hand, can be used to test fine-tuned models, which can be downloaded from their respective drive links (see the respective .txt files of models)


## Models
The fine-tuned models are "greedy" in the sense that they are trained on all the available data and evaluated solely based on cross-entropy loss. This stems from the fact that there is naturally no metric for evaluating the model's accuracy in terms of the task at hand. The evaluation of the model is done manually through certain test cases (1-word, 2-word, multiword, person/character name) in `predictor.py`
