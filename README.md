# XCTAB

## Python Package for Anomaly Detection in Wine Quality Tabular Data

<br>

## Structure tree
```
.         
├── data
│   ├── winequality.csv          
│   ├── ...            
│   ├── ...                      
│   └── (.csv files)    
├── logs
│   ├── ...            
│   ├── ...                      
│   └── (.log files)             
├── models
│   ├── ...            
│   ├── ...                      
│   └── (.pkl files)          
├── notebooks
│   └── xctab.ipynb
├── output                        
│   ├── ...            
│   ├── ...                      
│   └── (.csv & .png files)
├── samples
│   ├── red-sample.csv                       
│   └── red-samples.csv    
├── tests
│   ├── __init__.py                       
│   └── test_utilities.py            
├── xctab           
│   ├── __init__.py                
│   ├── config.py           
│   ├── data.py         
│   ├── test.py       
│   ├── train.py              
│   └── utils.py
├── README.md           
└── requirements.txt   
```

<br> 

## Setup

```
git clone https://github.com/edobytes/xctab.git
```

```
cd xctab
```

```
conda create --name xctab python=3.11 -y

conda activate xctab

pip install -r requirements.txt
```

<br>

## Usage

**Logging** and basic **CLI** have been implemented.

Call `--help` on `data.py`, `train.py`, and `test.py` to view the correct argument for said scripts. 

E.g.,

```
python ./xctab/train.py --help
```


### 1. Prepare data for ingestion

```
python ./xctab/data.py
```

### 2. Train a specified model on a specified train test 

```
python ./xctab/train.py [OPTIONS]
```

OPTIONS:

* `--wine-type` or `-wt` : _red_ or _white_
* `--model-name` or `-mn` : one of _autoencoder_, _ecod_, _knn_, _iforest_

E.g.,  `python ./xctab/train.py -wt red -mn knn`

Models' parameters are saved in `./models`.

### 3. Query the trained model either on the coresponding test set or on specified samples of the same wine type

```
python ./xctab/test.py [OPTIONS]
```

OPTIONS:
* `--wine-type` or `-wt` : _red_ or _white_
* `--model-name` or `-mn` : one of _autoencoder_, _ecod_, _knn_, _iforest_
* `--test-mode` or `-tm` : _prediction_ or _inference_
* `--export` or `-ex` : boolean

E.g., `python ./xctab/test.py -wt red -mn knn -tm inference --ex`

When using the `--export` flag all output (e.g., results of the query in .`csv` format and confusion matrices in `.png` format) are saved in `./output`

When querying _prediction_ on sample(s), the user will be prompted for a path to the `.csv` file containing the sample(s).

To test, one can use the two templates provided in `./samples`

E.g.,

`python ./xctab/test.py -wt red -mn knn -tm prediction --ex`

`"Type full path to sample(s) .csv file: "` (prompt)

`samples/red-samples.csv` (user input)
