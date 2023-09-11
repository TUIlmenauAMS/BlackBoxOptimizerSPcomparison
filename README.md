# BlackBoxOptimizerSPcomparison
A comparison of Black Box optimizers for source separation and RNN

This repository contains the software, models, and results text files for a comparison of different Black Box optimizers, including the Random Directions algorithm.

The main program are: 
- ICAabskl_puredelay_multichan_randdir_online.py for the source separation optimizers
- pytorch_rnn_IIR.py for the RNN optimizers

The optimizers can be chosen in the main programs, by uncommenting the appropriate line. The optimnization is run 10 times, and for each optimization run the obtained objective function or loss function value and needed run time are looged in the corresponding txt file.

For the text files in this repository, the hardware was a processor with 8 CPU cores, each running at 1.8 GHz. Observe that with Colab you will get somewhat different results because it usually just provides 2 cores. 
Hence it is recommended to let it run locally, but then it likely needs a virtual environment  for pypop7. A virtual environment can be created, for instance, with

pip install virtualenv

virtualenv pypop7

#activate:

source ~/pypop7/bin/activate

pip install pypop7

#deactivate with:

deactivate

To let the jupyter notebook "sourceSeprationBlackBoxComp.ipynb" run in Colab is more convenient, because it is its own envirnmonent and installs everything needed, but it is usually slower.

Gerald Schuller, September 2023
