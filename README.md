# popbias

research submitted to recsys 2022

Last updated 2022/05/27



# 0. Preprocess data

The preprocessing steps for each data is inside the **data** folder.

You should go to each of the folder and generate the **train_samples** for each dataset.

The **train_samples** is generated so that the training data for each epoch is deterministic, helping reproducibility.

Due to constraint in size, we only provide the rawdata for the **movielens**, and the **synthetic data**.

However, we provide the download link for the **gowalla, goodreads, ciao dataset**.

The preprocessing steps for the **gowalla, goodreads, ciao** dataset is equivalent as that of **movielens**.   





# 1. Run code


For synthetic experiments

```
!python main_basic_synthetic.py --dataset synthetic3 --model MF --sample "[none, pos2neg2, posneg, ipw, pd, pearson]"
```

For synthetic experiments - augmented data

```
!python main_basic_synthetic.py --dataset synthetic4 --model MF --sample "[none, pos2neg2, posneg, ipw, pd, pearson]"
```

For benchmark data experiments

note, separate scripts are used for MF, NCF models, and NGCF, LightGCN models

```
!python main_basic.py --model "[MF, NCF]" --dataset "[movielens, gowalla, goodreads, ciao] --sample "[none, pos2neg2, posneg, ipw, pd, macr, pearson] --weight [ex 0.9]"

!python main_graph.py --model "[NGCF, LightGCN]" --dataset "[movielens, gowalla, goodreads, ciao]" --sample "[none, pos2neg2, posneg, ipw, pd, macr, pearson] --weight 0.9"
```   

Note the --sample none correspond to the baseline BPR loss and the --sample posneg correspons to the zerosum method.

The --weight [] parameter controls the weight. It works differently for each method.

For pos2neg2, posneg, it controls the weight ratio between the BPR loss term (accuracy loss) and the regularization term (pop loss).

For instance, --sample posneg --weight 0.9 means the accuracy loss has 0.9/1.0 weight, and the popularity loss has 0.1/1.0 weight.

For pearson method, the --weight [] controls the absolute weight of the pearson regularization term.

The none, ipw method do not need the --weight term

The hyperparameter of the macr method is tuned after training, hence the --weight method is not needed.

For the pd method the --weight [] controls the popularity factor divided from the predicted item score.




# 2. Results

The trained MF, NCF, NGCF, LightGCN pytorch models are in the **models** folder.

The csv file of the training result is in the **experiments** folder.   

We provide the analysis results in the **research notebook -** folders. The analysis result for the **movielens** and **synthetic data** is provided.


# References

For the code of MF, NCF model, we modified by the codes provided from the following repository.    
https://github.com/guoyang9/NCF

For the NGCF, LightGCN model, we modified by the codes provided from the following repository.    
https://github.com/huangtinglin/NGCF-PyTorch

For the implementation of the Bayesian Pairwise Ranking Loss and related dataset construction, we modified by the codes provided from the following repository.    
https://github.com/guoyang9/BPR-pytorch


$\phantom{a}$


The repository will be continuously updated. Contact us for further questions or issues regarding the code.   

Thank you.
