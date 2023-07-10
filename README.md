# sWords-mt5pl2en
Automatic data annotator powered by mt5-pl2en translation model. The fine-tuned models try to to translate Polish key-meanings into English while also extracting keywords.

## Scripts
`fine_tuner_greedy.py` demonstrates the training script of greedy models, while `fine_tuner_conservative.py` demonstrates the script for conservative ones.


## Greedy models
Greedy models are trained on all the available data and evaluated solely based on cross-entropy loss. 

## Conservative models
Greedy models are trained on 80% of the available data and evaluated based on other metrics as well, such as BLEU.
