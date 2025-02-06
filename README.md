# Spell-Checker Fine-Tuning and Knowledge Distillation

This repository provides sample code for fine-tuning a spell checker based on Neuspell using a customized dataset. To develop a custom spell checker, we first need to generate noisy data from our dataset.

Follow the steps below:

### Dataset
The [adding_noise](adding_noise.py) script generates noisy data using the ProbabilisticCharacterReplacementNoiser method. Neuspell also provides additional options for noise generation.

## Fine-Tuning
The notebook [finetune_neuspell](finetune_neuspell.ipynb) contains code for fine-tuning Neuspell on the synthesized noisy dataset.

## Knowledge Distillation
The notebook [Knowledge_distillation_neuspell](Knowledge_distillation_neuspell.ipynb) provides code for knowledge distillation to create smaller BERT-based models, such as google/bert_uncased_L-2_H-256_A-4, which are more efficient for production use.


