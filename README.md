# BlackBoxOptimizerSPcomparison
A comparison of Black Box optimizers for source separation and RNN

This repository contains the software, models, and results text files for a comparison of different Black Box optimizers, including the Random Directions algorithm.

The main program are: 
- ICAabskl_puredelay_multichan_randdir_online.py for the source separation optimizers
- pytorch_rnn_IIR.py for the RNN optimizers

The optimizers can be chosen in the main programs, by uncommenting the appropriate line. The optimnization is run 10 times, and for each optimization run the obtained objective function or loss function value and needed run time are looged in the corresponding txt file.

For the text files in this repository, the hardware was a processor with 8 CPU cores, each running at 1.8 GHz. Observe that with Colab you will get somewhat different results because it usually just provides 2 cores.

Gerald Schuller, September 2023
