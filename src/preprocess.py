import pandas as pd
import  yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["preprocessing"]["root_dir"]
save_path=config['preprocessing']['source_dir']
print("Dataset path from config:", data_path)

df=pd.read_csv(data_path)
print(df.columns)

df.drop(columns=['Health_Issues','ID','Occupation','Sleep_Quality','Gender','Country'],inplace=True)

df.to_csv(save_path)