


import pandas as pd



def model_data(data):

    binary_cols = ['Street', 'Utilities', 'CentralAir']

    # Binary Encoding 
    data['Street'] = data['Street'].replace( {'PAVE': 1, 'GRVL': 0})
    data['Utilities'] = data['Utilities'].replace( {'ALLPUB': 1, 'NOSEWA': 0})
    data['CentralAir'] = data['CentralAir'].replace( {'Y': 1, 'N': 0})
    
    data[binary_cols] = data[binary_cols].astype(int)

    #Identifying nominal features
    nominal_cols = ['MSZoning', 'LandContour', 'HouseStyle', 'Electrical']

    # One-hot encoding for nominal columns
    import pickle 
    onehot = pickle.load(open('onehot_houseprice.pkl', 'rb'))

    encoded = onehot.transform(data[nominal_cols]).toarray()  # type: ignore
    
    data= data.drop(nominal_cols, axis=1)

    # Create a DataFrame with the encoded columns
    encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(), index=data.index)

    data = pd.concat([encoded_df, data], axis=1)

    # Scale the data by loading the scaler pickle file    
    min_max_scale = pickle.load(open('minmax_scaler_houseprice.pkl', 'rb'))

    data = pd.DataFrame(min_max_scale.transform(data), columns=data.columns, index=data.index)


    return data
