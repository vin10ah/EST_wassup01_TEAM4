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

### How to Run
1) `pip install -r requirements.txt` to install required packages
2) Set the mode in the pipeline model's config
3) you can choose one between two options : split = train, test, predict_mode = one_step, dynamic
4) also choose select_channel_idx = <br> # 0, # elec_amount (This must be used) <br>
                                      # 1, # temp<br>
                                      # 2, # wind_speed<br>
                                      # 3, # humidity<br>
                                      # 4, # rainfall<br>
                                      # 5, # sunshine<br>
                                      # 6, # rolling_mean<br>
                                      # 7, # diff<br>
5) then, try this!

### Dataset
- Source: (https://dacon.io/competitions/official/235736/data)
- Train & Test Set
	- total data : 2040
	- Train Set: 1681
 	- validation Set: 168
	- Test Set: 168
- Features
	- 7 features are selected 5 from over 9 features <br>
	(temp, wind_speed, humidity, rainfall, sunshine) <br>
	+ 2 features is created (rolling mean, diff)
  
- Target
	- elec_amout

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
