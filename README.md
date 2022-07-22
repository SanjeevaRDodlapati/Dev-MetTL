# Pub-Met-TL
This repository provides code for implementation of [Completing Single-Cell DNA Methylome Profiles via Transfer Learning Together With KL-Divergence](https://www.frontiersin.org/articles/10.3389/fgene.2022.910439/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Genetics&id=910439)

Installation
============

Clone the repository

```
git clone https://github.com/ODU-CSM/Pub-Met-TL.git
```



and then install Met-TL using ``setup.py``:

```
  python setup.py install
```





Training
===============
Train your own model from scratch or fine-tune pretrained model with ``train.py``:
```
  train.py
    ./examples/input/c{1,4,7,10,13,16,19,22}_*.h5
    --val_data ./examples/input//c{3,6,9,12,15,18,21}_*.h5
    --dna_model SeqCnnL2h128
    --cpg_model MetRnnL1
    --joint_model JointL2h512
    --nb_epoch 30
    --out_dir ./examples/train
```



This command uses chromosomes 1,4,7,10,13,16,19,22 for training and 3,6,9,12,15,18,21 for validation. ``---dna_model``, ``--cpg_model``, and ``--joint_model`` specify the architecture of the Met, Sequence, and Joint model, respectively (see manuscript for details). Training will stop after at most 30 epochs and model files will be stored in ``./train``.



Prediction
===============
Use ``eval.py`` to impute methylation profiles and evaluate model performances.

```
  eval.py
    ./data/*.h5
    --model_files ./model/model.json ./model/model_weights_val.h5
    --out_data ./eval/data.h5
    --out_report ./eval/report.tsv

```


This command predicts missing methylation states on all chromosomes and evaluates prediction performances using known methylation states. Predicted states will be stored in ``./examples/eval/data.h5`` and performance metrics in ``./examples/eval/report.tsv``.





Note: This code is developed from adopting parts of the code from [deepcpg](https://github.com/cangermueller/deepcpg)


Contact
=======
* Sanjeeva Dodlapati
* sdodl001@odu.edu
* https://sdodlapati.com
* [@dodlapati_reddy](https://twitter.com/dodlapati_reddy)
