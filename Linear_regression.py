import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text


class predicting:
    def __init__ (self):
        a = 0

    #Данные функции были созданы для того, чтобы преобразовать изначальный файл, который был предоставлен, в более удобный формат.   
    
    # В последующих версиях, она необязательна для использования, поскольку модель будет извлекать данные прямо из базы данных.
    def preparation(self):        
        df = pd.read_excel(r"C:\Users\user\OneDrive\Рабочий стол\Data Science\Стажировка\Линейная регрессия\Данные_по_рынку_недвижимости_powerbi_формат.xlsx", sheet_name = 'Сводная файл')
        df = df.drop("№", axis = 1)
        df = df[df['Статус'] == 'Факт']
        df = df.dropna()
        df.rename(columns={'Область/Город': 'Регион'}, inplace=True)
        df.rename(columns={'Название информации': 'Информация'}, inplace=True)
        return df

    #Загружает преобразованные данные в таблицу
    def inserting(df):
        engine = create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/данные_по_рынку_недвижимости", echo=True)
        conn = engine.connect()
        #Вручную создаем таблицы, чтобы правильно задать типы данных столбцов
        conn.execute(text("""                
        CREATE TABLE IF NOT EXISTS Data(
            `№` int auto_increment,
            `Область/Город` TEXT, 
            Год INT, 
            `Название информации` TEXT,
            Значение FLOAT, 
            Статус TEXT       
            )
        """))
        conn.commit()
        df.to_sql('data', engine, if_exists = 'append',  index=False)


#-----------------------------------------------------------------------------------------------------------
    

    #Простая функция, которая извлекат данные из базы данных, благодаря SQLalchemy
    def extract_data(self):
        engine = create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/данные_по_рынку_недвижимости", echo=True)
        conn = engine.connect()
        df = pd.read_sql('data', con = conn)
        return df
        
    
    def linear_regression(self, df ):
        indicators = df['Информация'].unique()
        regions = df['Регион'].unique()
        pv_df = pd.pivot_table(df, values = "Значение", index = ["Регион", "Год"], columns = "Информация") # Создаем сводную таблицу, чтобы легко доставать данные по регионам и годам
        pv_df = pv_df.interpolate(method = 'linear') # Заполнение пропусков, методом интерполяции
        pv_df = pv_df.dropna() #Некоторые ряды имеют слишком много пропусков и их уже приходится удалять
        print(pv_df)
        print('='*50)
        region_tables = {}
        for region in regions:
            try:
                region_data = pv_df.xs(region, level='Регион') # Достаем данные регионов из мультииндекса
                region_tables[region] = region_data
            except KeyError:
                print(f"Нет данных для региона: {region}")
                continue
        forecast_years = [2025, 2026, 2027, 2028, 2029, 2030, 2031]
        forecast_df = pd.DataFrame()
        for region in regions:
            region_data = pv_df.xs(region, level='Регион')
            for indicator in indicators:
                 X = region_data.index.get_level_values('Год').values.reshape(-1, 1)
                 y = region_data[indicator].values
         
                 model = LinearRegression()
                 model.fit(X, y)
         
                 future_years = np.array(forecast_years).reshape(-1, 1)
                 future_predictions = model.predict(future_years)

                 for year, pred in zip(forecast_years, future_predictions):
                     forecast_df = pd.concat([
                        forecast_df,
                        pd.DataFrame({
                             'Регион': [region],
                             'Год': [year],
                             'Информация': [indicator],
                             'Значение': [pred],
                             'Статус': 'Прогноз'
                         })
                     ], ignore_index=True)
        return forecast_df

    # Функция для визуализации всех предсказаний, вместе с настоящими данными 
    def visuals(self, df, forecast_df ):
        indicators = df['Информация'].unique()
        regions = df['Регион'].unique()
        final_df = pd.concat([forecast_df, df])
        pv_df1 = pd.pivot_table(final_df, values = "Значение", index = ["Регион", "Год"], columns = "Информация")

        for region in regions:
            region_data = pv_df1.xs(region, level='Регион')
            for indicator in indicators:
                plt.plot(region_data.index.get_level_values('Год').values.reshape(-1, 1), region_data[indicator].values, 
                    marker='o', 
                    linewidth=2,
                    markersize=8,
                    label=indicator)
                plt.title(region, fontsize=16, pad=20)
                plt.xlabel('Год', fontsize=12)
                plt.ylabel(indicator, fontsize=12)
                plt.xticks(rotation=45, fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True, alpha=0.3)


                plt.tight_layout()
                plt.show()

    #Функция, которая позволяет вставить данные предсказаний обратно в базу данных 
    def inserting_predictions(self, forecast_df):
        engine = create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/данные_по_рынку_недвижимости", echo=True)
        conn = engine.connect()
        forecast_df.to_sql('data', engine, if_exists = 'append',  index=False)


pre = predicting()
df = pre.extract_data()
forecast_df = pre.linear_regression(df)

# pre.visuals(df, forecast_df) 
# Программе нужно визуализировать каждый индикатор каждого региона по отдельности, поэтому я не буду ее запускать
# Вы можете сами запустить, если хотите взглянуть на предсказания и сравнить их.

pre.inserting_predictions(forecast_df)