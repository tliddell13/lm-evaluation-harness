## This project is an extension of the original lm-evaluation-harness. 
This version allows for permutations to be carried out on the datasets during evaluation.
So that the experiments in my paper can be easily repeated permutations can be added as an argument 
when running the main.py file.

Permutations that can be applied when running:
shuffle=None, - Shuffles question. Can be unigram, bigram, or trigram shuffle
shuffleAnswer=None, - Can be unigram, bigram, or trigram shuffle
remove_question=False, - This argument allows the question to be 
posReplace=None, - This argument replaces a pos with its synonym
extra_answers=False, - Has a language model generate a distracting answer to be added to the choices
named_entities=None, - Remove all, or keep only named entities
cot = False, - Use chain of thought prompting for GSM8K

Some examples of this can be seen as Slurm jobs in the SlurmEvals folder.
The functions for these are defined in permutations.py and applied in evaluator.py

## Results
The results for these studies are examined in the permutation_results folder which contains outputs 
and testing accuracies for the models. 
