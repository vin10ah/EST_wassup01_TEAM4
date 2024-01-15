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
	+ 2 features are created (rolling mean, diff)
  
- Target
	- elec_amout


### Results



![24](https://github.com/Bong-HoonLee/EST_wassup01_TEAM4/assets/144428051/25f9bae9-2630-4047-90da-234bddd3f346)

![168](https://github.com/Bong-HoonLee/EST_wassup01_TEAM4/assets/144428051/84ed751c-0870-4552-a77e-24f252c0b22d)
