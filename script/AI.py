import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s [line %(lineno)d]')


class LOAD:
    def __init__(self, file_path):
        self.filepath = file_path


class EDA(LOAD):

    def cleanfile(self):
        self.filee = None 
        
        if self.filepath is not None:
            try:
                if os.path.isfile(self.filepath):
                    self.filee = pd.read_csv(self.filepath)
                    print(self.filee.head())
                    print(self.filee.info())
                    print(self.filee.isnull().any())
                    print(self.filee.duplicated().any())
                    print(self.filee.columns)
                    print(self.filee.dtypes)

                    if self.filee.duplicated().sum()>0:
                        self.filee.drop_duplicates(keep="first",inplace=True)
                        self.filee.reset_index(drop=True,inplace=True)

                    if self.filee.isnull().sum().sum()>0:
                        self.filee.dropna(inplace=True)
                    
                    return self.filee
                
                else:
                    logging.error("The specified path is not a file.")
                    
            except Exception as e:
                logging.error(e)
        else:
            logging.error("File path is None or invalid.")       
            
            
    def analysis(self):
        
        if self.filee is not None:
            
            try:
                print(self.filee.describe())
                print(self.filee["High"].mean())

                if 'Date' in self.filee.columns:
                    self.filee['Date']=pd.to_datetime(self.filee['Date']).dt.year
                else:
                    raise Exception

                print(self.filee.head())

                gr1 = self.filee.groupby("Date").agg({"High": "max", "Low": "min"})
                print(gr1)
                
                gr2 = self.filee.groupby("Date").agg({"Open": "max", "Close": "min"})
                print(gr2)
                
                gr3= self.filee.groupby("Date")["Volume"].mean().reset_index()
                print(gr3)
                
                return gr1,gr2
        
            except Exception as e:
                logging.error(e)
        
        else:
         logging.error("File is Empty.......")
    
         
    def savefile(self):
        if self.filee is not None:
         try:
            self.filee.to_csv("bitcoin_mod.csv",index=False)
            print("file saved...........")
         except Exception as e:
            logging.error(e)
        else:
            logging.error("file not found.........")
            
            
    def visualize1(self, gr1,gr2):
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1,2,1)
        plt.title("High and Low of Bitcoin (2014–2024)", fontsize=14)

        # Melt the DataFrame to long format for grouped barplot
        df_melted1 = gr1.reset_index().melt(id_vars='Date', value_vars=['High', 'Low'],var_name='Price Type', value_name='Value')

        sns.barplot(x='Date', y='Value', hue='Price Type', data=df_melted1,palette="viridis")
        
        plt.subplot(1,2,2)
        
        plt.title("Open and Close of Bitcoin (2014–2024)", fontsize=14)
        df_melted2 = gr2.reset_index().melt(id_vars='Date', value_vars=['Open', 'Close'],var_name='Price Type', value_name='Value')
        sns.barplot(x='Date', y='Value', hue='Price Type', data=df_melted2,palette="winter")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        plt.title("Average Volume (2014–2024)", fontsize=14)
        sns.lineplot(x=self.filee["Date"],y=self.filee["Volume"],palette="winter",linewidth=5)
        plt.tight_layout()
        plt.show()
        
        plt.title("Correlations", fontsize=14)
        sns.heatmap(self.filee.corr(numeric_only=True),annot=True, cmap="coolwarm", linewidths=0.001)
        plt.tight_layout()
        plt.show()
        
         
class ML(EDA):
    def ml(self):
        df = self.filee.copy()

        X1 = df.drop(columns=["Date", "High"], axis=1)
        Y1 = df["High"]
        X2 = df.drop(columns=["Date", "Low"], axis=1)
        Y2 = df["Low"]

        x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, random_state=42, train_size=0.7)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, random_state=42, train_size=0.7)
        

        models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor()
        }

        for name, model in models.items():
            print(f"\n====== {name} ======")
            
            

            model.fit(x_train1, y_train1)
            preds1 = model.predict(x_test1)
            mae1 = mean_absolute_error(y_test1, preds1)
            mse1 = mean_squared_error(y_test1, preds1)
            r2_1 = r2_score(y_test1, preds1)

            model.fit(x_train2, y_train2)
            preds2 = model.predict(x_test2)
            mae2 = mean_absolute_error(y_test2, preds2)
            mse2 = mean_squared_error(y_test2, preds2)
            r2_2 = r2_score(y_test2, preds2)

            print("--------- FOR HIGH ----------")
            print(f"MAE: {mae1:.4f}")
            print(f"MSE: {mse1:,.2e}")
            print(f"R² Score: {r2_1:.4f}")

            print("--------- FOR LOW -----------")
            print(f"MAE: {mae2:.4f}")
            print(f"MSE: {mse2:,.2e}")
            print(f"R² Score: {r2_2:.4f}")

            comparison = pd.DataFrame({
                'Adj Close': x_test1['Adj Close'].values,
                'Close': x_test1['Close'].values,
                'Open': x_test1['Open'].values,
                'Volume': x_test1['Volume'].values,
                'High(actual)': y_test1.values,
                'High(predicted)': preds1,
                'Low(actual)': y_test2.values,
                'Low(predicted)': preds2
            })

            print("\nSample Predictions:")
            print(comparison.head(10))
            
            
            plt.figure(figsize=(15, 6))
            plt.plot(y_test1.values, label='Actual High',marker='o')
            plt.plot(preds1, label='Predicted High',marker='o')
            plt.title(f"{name} - High Price Prediction")
            plt.xlabel("Samples")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            
            plt.figure(figsize=(15, 6))
            plt.plot(y_test2.values, label='Actual Low', marker='o')
            plt.plot(preds2, label='Predicted Low', marker='x')
            plt.title(f"{name} - Low Price Prediction")
            plt.xlabel("Samples")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()           
            

b=ML(file_path=r"Bitcoin_Historical_Data.csv")
print("***********CLEANING FILE**************")
b.cleanfile()
print("***********ANALYSIS**************")
gr1,gr2=b.analysis()
print("***********VISUALIZATION**************")
# b.visualize1(gr1,gr2)
# print("***********SAVING FILE**************")
# b.savefile()
print("*************Predicting Model************")
b.ml()


