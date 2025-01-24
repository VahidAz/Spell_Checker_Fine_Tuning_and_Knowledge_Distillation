# This script is for generating noisy data to train spell checker models


# Install Libs
import subprocess
def run(cmd):
    print(subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8"))

run("git clone https://github.com/neuspell/neuspell")
run('pip install -e ./neuspell/.')
run("pip install unidecode")
run("pip install -U spacy")
run("pip install -U elmo")
run("python -m spacy download en_core_web_sm")


# Headers
from neuspell.noising import ProbabilisticCharacterReplacementNoiser
import pickle as plk


# Load the data
# The data is a list of sentences extracted from your docs
with open("PATH_TO_YOUR_DATA", "rb") as fin:
    sentences = plk.load(fin)


# Output files
fcorrect = open("CORRECT_SENTENCES.txt", "a")
fnoise = open("NOISY_SENTENCES.txt", "a")


# Load the noise generators from Neuspell
my_noiser = ProbabilisticCharacterReplacementNoiser(language="english")
my_noiser.load_resources()
preprocessor = ProbabilisticCharacterReplacementNoiser.create_preprocessor(lower_case=True, remove_accents=True)
retokenizer1 = ProbabilisticCharacterReplacementNoiser.create_retokenizer()
retokenizer2 = ProbabilisticCharacterReplacementNoiser.create_retokenizer(use_spacy_retokenization=True)


# Iterating over sentences and generating noisy sentences with three different 'CharacterReplacementNoisers'
for count, sen in enumerate(sentences[:]):
    noise_texts = my_noiser.noise(sen)
    noise_texts = [' ' if x=='' else x for x in noise_texts]
    noise_texts = ''.join(noise_texts)
    noise_texts = noise_texts.lower()
    
    if not sen.__eq__(noise_texts):
        fcorrect.write(sen)
        fcorrect.write("\n")

        fnoise.write(noise_texts)
        fnoise.write("\n")
    
    noise_texts = my_noiser.noise(sen, preprocessor=preprocessor, retokenizer=retokenizer1)
    noise_texts = [' ' if x=='' else x for x in noise_texts]
    noise_texts = ''.join(noise_texts)
    noise_texts = noise_texts.lower()
    
    if not sen.__eq__(noise_texts):
        fcorrect.write(sen)
        fcorrect.write("\n")

        fnoise.write(noise_texts)
        fnoise.write("\n")
    
    noise_texts = my_noiser.noise(sen, preprocessor=preprocessor, retokenizer=retokenizer2)
    noise_texts = [' ' if x=='' else x for x in noise_texts]
    noise_texts = ''.join(noise_texts)
    noise_texts = noise_texts.lower()
    
    if not sen.__eq__(noise_texts):
        fcorrect.write(sen)
        fcorrect.write("\n")

        fnoise.write(noise_texts)
        fnoise.write("\n")


fcorrect.close() 
fnoise.close()