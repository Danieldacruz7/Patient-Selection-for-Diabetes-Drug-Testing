import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    new_ndc_df = ndc_df.copy()
    new_ndc_df.drop(['Non-proprietary Name', 'Dosage Form', 'Route Name', 'Company Name', 'Product Type'], axis=1, inplace=True)
    df = pd.merge(df, new_ndc_df, how='inner', left_on='ndc_code', right_on='NDC_Code', copy=True)
    df.rename(columns = {'Proprietary Name' : 'generic_drug_name'}, inplace = True)
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    first_encounter_df = pd.read_csv("first_encounter_df.csv") # Dataframe created and saved to speed up notebook
    
    #df_columns = df.columns
    #df = df.sort_values('encounter_id', ascending=True)
    #unique_ids = df['patient_nbr'].unique()
    
    #first_encounter_df = pd.DataFrame(columns=df.columns)
    #new_df = pd.DataFrame(columns=df.columns)
    
    #for i in unique_ids:
    #    df1 = pd.DataFrame(df[df['patient_nbr'] == i].iloc[0]).T
    #    first_encounter_df = pd.concat([first_encounter_df, df1], copy=True, axis = 0)
        
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    test_percentage = 0.2
    valid_percentage = 0.1
    
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * (1 - test_percentage - valid_percentage))
    test_size = round(total_values * (test_percentage))
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[sample_size:sample_size+test_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[sample_size+test_size:])].reset_index(drop=True)
    
    print(total_values)
    
    print("Total number of unique patients in train = ", len(train[patient_key].unique()))
    print("Total number of unique patients in test = ", len(test[patient_key].unique()))
    print("Total number of unique patients in validation = ", len(validation[patient_key].unique()))
    print("Training partition has a shape = ", train.shape) 
    print("Test partition has a shape = ", test.shape)
    print("Test partition has a shape = ", validation.shape)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path)        
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)  
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=0, dtype=tf.float64, normalizer_fn=(lambda x: (x-MEAN)/STD))
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x>=5 else 0 )
    return student_binary_prediction
