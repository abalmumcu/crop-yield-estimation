from utils.preprocessing import DataProcessing

class Case:
    def __init__(self,day,dynamic_drop, static_drop):
        self.total_dataset_number = 5
        self.selected_day = day
        self.drop_dynamic_col_num = len(dynamic_drop)
        self.drop_feature_list = dynamic_drop + static_drop
        self.dynamic_col_num = 13

    def case_preprocessing (self,datasets,dataset_dict,target,concat = False, vector_df_old = None):
        import pandas as pd
        """
                dfs is [full_df,df_dropped,vector_df]\n
                scaler is [ScalerX and ScalerY(yield also included)]\n
                train_test is scaled [X_train, X_test, y_train, y_test, yield]

        """
        dfs = []
        preprocess = DataProcessing()

        full_df = preprocess.combine_datasets(datasets,
                                          dataset_dict,
                                          total_dataset_number=self.total_dataset_number, 
                                          selected_day=self.selected_day)
        
        df_dropped = preprocess.drop_feature(full_df,self.drop_feature_list)
        vector_df, _ = preprocess.vectorize_dataset(df_dropped,
                                                   self.dynamic_col_num - self.drop_dynamic_col_num + 1) # + 1 for Date
        preprocess.check_na(vector_df)
        if concat ==True:
            vector_df = pd.concat([vector_df_old,vector_df],axis=0)
        dfs.append([full_df,df_dropped,vector_df])
        scaler, train_test = preprocess.train_test_split_wscaler(target,vector_df)

        return dfs, scaler, train_test