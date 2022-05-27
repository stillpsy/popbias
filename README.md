# popbias

research submitted to recsys 2022



0. Preprocess data

The preprocessing steps for each data is inside the **data** folder.

You should go to each of the folder and generate the **train_samples** for each dataset.

The **train_samples** is generated so that the training data for each epoch is deterministic, helping reproducibility.

Due to constraint in size, we only provide the rawdata for the **movielens**, and the **synthetic data**.

However, we provide the download link for the **gowalla, goodreads, ciao dataset**.

The preprocessing steps for the **gowalla, goodreads, ciao** dataset is equivalent as that of **movielens**.





1. Run code


For synthetic experiments

python main_basic_synthetic.py --dataset synthetic3 --model MF --sample "[none, pos2neg2, posneg, ipw, pd, pearson]"


For synthetic experiments - augmented data

python main_basic_synthetic.py --dataset synthetic4 --model MF --sample "[none, pos2neg2, posneg, ipw, pd, pearson]"


For benchmark data experiments

python main_basic.py --model "[MF, NCF]" --dataset "[movielens, gowalla, goodreads, ciao]"

python main_graph.py --model "[NGCF, LightGCN]" --dataset "[movielens, gowalla, goodreads, ciao]"



2. Results

The trained MF, NCF, NGCF, LightGCN pytorch models are in the **models** folder.

The csv file of the training result is in the **experiments** folder.


3. We provide the analysis results in the **research notebook -** folders.


The repository will be continuously updated. Contact us for further questions regarding the code.

