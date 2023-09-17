import numpy as np
import pandas as pd

def data_preprocessing(dataset_dict_tmp,datasets_tmp,selected_day=15):
    df_list_tmp = []
    yield_list_tmp = []
    evi_list_tmp = []
    raw_list_tmp = []
    i = 0
 
    #scaler_full = Robust_scale()

    for state_tmp in dataset_dict_tmp.keys():
        for city_tmp in dataset_dict_tmp[state_tmp]:
            for dataset_number_tmp in range(5):
                    df_all_tmp = datasets_tmp[dataset_number_tmp][state_tmp][city_tmp]['dynamic']
                    yield_data_tmp = datasets_tmp[dataset_number_tmp][state_tmp][city_tmp]['yield']
                    df_all_tmp.index = pd.to_datetime(df_all_tmp.index)
                    df_all_tmp = df_all_tmp.sort_index(ascending=True)
                    df_all_tmp = df_all_tmp.drop('lst_day', axis=1)
                    try:
                        df_all_tmp = df_all_tmp.drop('yield', axis=1)
                        for idx_tmp in range(1,12):
                            df_all_tmp = df_all_tmp.drop(f'sf_{idx_tmp}',axis=1)
                    except:
                        pass
                    df_all_tmp = df_all_tmp.drop('lst_nigth', axis=1)
                    df_all_tmp = df_all_tmp.drop('dayl', axis=1)


                    df_all_tmp = df_all_tmp.interpolate()
                    df_monthly_tmp = df_all_tmp[(df_all_tmp.index.month >= 4) & (df_all_tmp.index.month <= 9)]
                    
                    #df_dayly_tmp = df_monthly_tmp.rolling('7D', center=True).mean()
                    
                    window_size_tmp = 7  # Length of the moving average filter
                    filter_radius_tmp = window_size_tmp // 2  # Radius of the filter (number of days before and after the current day)

                    filtered_df_tmp = pd.DataFrame()  # DataFrame to store the filtered data

                    # Apply moving average filter to each feature_tmp
                    for feature_tmp in df_monthly_tmp.columns:
                        moving_averages_tmp = []
                        for i in range(len(df_monthly_tmp)):
                            start_idx_tmp = max(i - filter_radius_tmp, 0)
                            end_idx_tmp = min(i + filter_radius_tmp + 1, len(df_monthly_tmp))
                            window_tmp = df_monthly_tmp.iloc[start_idx_tmp:end_idx_tmp][feature_tmp]
                            window_average_tmp = round(np.mean(window_tmp), 6)
                            moving_averages_tmp.append(window_average_tmp)

                        filtered_df_tmp[feature_tmp] = moving_averages_tmp

                    #filtered_df_tmp['date'] = pd.to_datetime(df_monthly_tmp.index)

                    # Set the 'date' column as the dataframe's index
                    filtered_df_tmp.set_index(df_monthly_tmp.index, inplace=True)
                
                    df_dayly_tmp = filtered_df_tmp[(filtered_df_tmp.index.day == selected_day)]


                    df_dayly_tmp.index = pd.to_datetime(df_dayly_tmp.index, format = '%Y-%m-%d').strftime('%m-%d')
                    df_dayly_tmp['evi'] = df_dayly_tmp['evi'].astype(np.float64)
                    df_dayly_tmp['lai'] = df_dayly_tmp['lai'].astype(np.float64)
                    df_dayly_tmp['fpar'] = df_dayly_tmp['fpar'].astype(np.float64)
                    df_dayly_tmp['ssm'] = df_dayly_tmp['ssm'].astype(np.float64)
                    df_dayly_tmp['susm'] = df_dayly_tmp['susm'].astype(np.float64)
                    
                    if all(0 <= x <= 1 for x in df_dayly_tmp.iloc[:, 0]): #and ((max(df_dayly_tmp.iloc[:,0]) == df_dayly_tmp.iloc[4,0])):# or (max(df_dayly_tmp.iloc[:,0]) == df_dayly_tmp.iloc[3,0])) :
                        
                        
                    
                        df_filtered_tmp = df_dayly_tmp.select_dtypes(include=['float64', 'int64'])  
                        #scaler = Ro()
                        #scaler.fit(df_filtered_tmp)
                        #df_scaled = pd.DataFrame(scaler.transform(df_filtered_tmp), columns=df_filtered_tmp.columns)

                        # Scale the input data

                        raw_list_tmp.append(df_filtered_tmp)
                        df_list_tmp.append(df_filtered_tmp)
                        yield_list_tmp.append(yield_data_tmp)
                        evi_list_tmp.append(df_filtered_tmp.evi)
                        
                        
    df_full_tmp = np.stack(df_list_tmp)
    yield_full_tmp = np.stack(yield_list_tmp)
    evi_full_tmp = np.stack(evi_list_tmp)
    raw_full_tmp = np.stack(raw_list_tmp)
        
    return df_full_tmp,yield_full_tmp,evi_full_tmp,raw_full_tmp,df_filtered_tmp

