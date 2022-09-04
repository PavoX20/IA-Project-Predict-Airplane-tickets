import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
if __name__=="__main__":
  def predecir(airline,	source,	destination,	stops,	month,	day,	duration):
    knn_model = pickle.load(open("/Users/pavox20/Library/CloudStorage/OneDrive-EscuelaPolitécnicaNacional/Semestres/7mo Semestre/Inteligencia Artifical/Ejercicios Base de datos/Proyecto/KNN.sav", 'rb'))
    loaded_model = pickle.load(open("/Users/pavox20/Library/CloudStorage/OneDrive-EscuelaPolitécnicaNacional/Semestres/7mo Semestre/Inteligencia Artifical/Ejercicios Base de datos/Proyecto/SVM.sav", 'rb'))
    modelo=load_model("/Users/pavox20/Library/CloudStorage/OneDrive-EscuelaPolitécnicaNacional/Semestres/7mo Semestre/Inteligencia Artifical/Ejercicios Base de datos/Proyecto/ANN.h5")
    ticket= {
    'model_name':[airline],
    'transmission' :[source],
    'odometer_value':[destination],
    'year_produced':[stops],
    'engine_capacity':[month],
    'body_type':[day],
    'has_warranty':[duration]
    
      }
    
    dataframe = pd.read_csv('/Users/pavox20/Library/CloudStorage/OneDrive-EscuelaPolitécnicaNacional/Semestres/7mo Semestre/Inteligencia Artifical/Ejercicios Base de datos/Proyecto/dataframe.csv', sep=";")
    
    X= dataframe
    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(X)
    X = minmaxscaler.transform(X)
    
    
    df = pd.DataFrame(ticket)
    
    
    ticket = df.iloc[0]
    ticket = minmaxscaler.transform(ticket.values.reshape(-1,7))

    print(knn_model.predict(ticket)[0])
    print(loaded_model.predict(ticket)[0])
    resultado= modelo.predict(ticket)
        
    return resultado[0][0]

    
datos=sys.argv

res=predecir(datos[1],datos[2],datos[3],datos[4],datos[5],datos[6],datos[7])
        

print(res)
    



   