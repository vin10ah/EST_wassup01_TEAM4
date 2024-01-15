### Team
- [BONGHOON LEE](https://github.com/Bong-HoonLee)
- [SEOBIN BANG](https://github.com/vin10ah)
- [DOYEON KIM](https://github.com/electronicguy97)
- [JAESEOK LEE](https://github.com/appleman153)
- [MOONSUN JUNG](https://github.com/JUNGMOONSUN/)

### Directory
- `archive`: history
- `data`: origin train data
- `config`: setting files for model parameters
- `doc`: documents, images, reports
- `pipeline`: ANN models, PATCHTST models
- `results`: Random Forest models
- `requirements.txt`: required libraries and packages
- 
### How to Run
1) `pip install -r requirements.txt` to install required packages
2) Set the mode in the pipeline model's config
3) you can choose one between two options : split = train, test, predict_mode = one_step, dynamic
4) also choose select_channel_idx = 0, # elec_amount (This must be used) <br>
                                      # 1, # temp<br>
                                      # 2, # wind_speed<br>
                                      # 3, # humidity<br>
                                      # 4, # rainfall<br>
                                      # 5, # sunshine<br>
                                      # 6, # rolling_mean<br>
                                      # 7, # diff<br>
5) then, try this!

### Dataset
- Source: 
- Train & Test Set
	- Train Set: survey of 2007 ~ 2019, and 2021 
	- Test Set: survey of 2020
- Features (independent variables, X)
	- 125 features are selected from over 800 features 
		- including the results of multiple blood tests, urine tests to intensity of daily workouts, education level, and even whether to brush their teeth.
	- Tried to use as many as possible features but,
		1) features with a high proportion of NaN values are discarded. 
		2) features which can be representative of similar questions
		3) commonly used features among 2007 ~ 2021, selected year.
- Target (dependent variable, y)
	- `depressed` variables have been defined. `depressed` == 1 if:
		1) `mh_PHQ_S` >= 10 or # `mh_PHQ_S`: total score of PHQ self test
		2) `BP_PHQ_9`.isin([1, 2, 3]) or # `BP_PHQ_9`: 9th question of PHQ self-test, "Have you ever thought about suicide or hurting yourself this year?"
		3) `BP6_10` == 1 or  # `BP6_10`: "Have you ever thought about suicide this year?"
		4) `BP6_31` == 1 or # `BP6_31`: "Have you ever thought about hurting yourself this year?"
		5) `DF2_pr` == 1 & `mh_PHQ_S`== NaN # `DF2_pr==1`: currently experiencing depression and have been diagnosed by a doctor
	- A respondent with no information on depression-related variables has been removed rather than filling in missing values.
	- Reference of definition of depression: National Health Insurance ([link](https://www.nhis.or.kr/static/alim/paper/oldpaper/202001/sub/s01_02.html))

### Preprocessing
![Pre_processing](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/54a9fb38-71f9-4a13-a8e5-7c28efd41477)
- Manual Processing
	- Data Loading, Train/Test Separation, Feature Selection and Grouping, Define Target Label, Drop NaNs -> output to csv
- In Pipeline
	- Fill NaNs (KNN, most frequent values), Encoding/Scaling, Under/Over-sampling


### Models
- Pre-processed data(.csv): [drive link](https://drive.google.com/drive/folders/1UjUa46Cx-X8-EDdWtWvQhg5gAbJgRlP3)
	- file name of the csv is corresponded with config.py
	- csv files should be located on `./data` 
- Metrics: Accuracy, Precision, Recall, F1-score, Support, AUROC
- Best Model (ANN)
  	- config_path: `./config/20231211_final/config_col_01_transformed.py`
  	- run settings: `./bin/train --mode=validate --config-dir=config/20231211_final`
  	- config settings
  	  	1) Model=ANN
		2) Model Params=ModuleList(
		 (0): Linear(in_features=208, out_features=2, bias=True)
		 (1): BatchNorm1d(2)
		 (2): ReLU()
		 (2): Linear(in_features=2, out_features=1, bias=True)
		3) loss_function=BCEwithLogitsLoss,
		4) optimizer=Adam,
		5) lr=0.001
		6) epochs=50
		7) batch size = 128
		8) lr scheduler = ReduceLROnPlateau
		   ( 'mode': 'min', 'factor': 0.1, 'patience': 5)
  	- results
  	 ![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/0bae431d-13b6-4ac4-a7b0-4cd51261fcca)
	 ![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/f31752fc-3f9e-4df0-a887-2290869b4fff)

- Comparison: Logistic Regression, Random Forest, ANN
- Detailed results with Korean can be found on /doc/20231212_report/20231212_report.ipynb

### Feature Importances
- Top 5 features based on random forest - feature importances
	- `D_1_1`: Subjective health awareness (0.0439)
	- `ainc`: Monthly average household total income (0.0263)
	- `LQ4_00_1.0`: Presence of (physical) activity limitation (Yes) (0.0240)
	- `edu`: Educational level - Academic background (0.0229)
	- `D_2_1_1.0`: Recent experience of physical discomfort in the past 2 weeks (Yes) (0.0214)
- Excluding any of the above features didn't show a significant change in metrics with Random Forest Model.
- Extra work is needed with the Neural Network Model to identify significant features.

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
