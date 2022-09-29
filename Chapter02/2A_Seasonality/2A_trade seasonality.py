import os
import pickle
import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


'''*************************************
Contants needed for the program
'''
data_path = os.getcwd()
os.chdir(data_path)
list_flds = []
fld_name = ''
index_col_name = 'Month'
file_path_out = os.path.join(data_path, 'output_dataset.txt')
f_name = sorted([item for item in os.listdir(data_path) if ('csv' in item)])

'''*************************************
    this loop import all data files, with dataframe name specified in the import_file_names specified
    input: raw data files in csv format, header starting at line 4; and import_file_names.txt
    output: dataframe list that store all dataframe names
'''


def shorten_name(name: str) -> str:
    concepts = ['average', 'cost', 'revenue', 'consumption']
    sectors = [
        'commercial', 'industrial', 'other', 
        'residential', 'transportation', 'all',
    ]
    fuels = [
        'coal', 'natural', 'gas', 
        'petroleum', 'coke', 'liquids',
    ]
    short_name = '_'.join([
        item.title() 
        for item in name.split('_') 
        if (item.lower() in (concepts + sectors + fuels))
    ])
    return short_name


def preprocess_data(filenames: List[str]) -> pd.DataFrame:
    dfs = []
    for name in filenames:
        short_name = shorten_name(name)
        df = (
            pd
            .read_csv(name, header=4)
            .set_axis(['Month', short_name], axis=1)
            .astype({'Month': 'datetime64[ns]'})
            .set_index('Month')
        )
        dfs.append(df)
    df2 =  pd.concat(dfs, axis=1).interpolate().fillna(0.0)
    return df2


'''*************************************
this function load data from a file path and create a monthly time index for the dataframe
'''


def load_data(file_path_out, fld_name, idx_start, idx_end):

    df_all = pd.read_csv(file_path_out, sep='\t')

    # Using Statsmodels...
    dta = df_all[[index_col_name, fld_name]]

    # seasonal difference
    dta_df = dta[fld_name]
    dta_df.index = pd.DatetimeIndex(start=idx_start, end=idx_end, freq='MS')
    return dta_df


def print_plt(data, fld):
    data.actual.plot(color='blue', grid=True, label='actual', title=fld)
    data.predicted.plot(color='red', grid=True, label='predicted')
    plt.legend()
    plt.savefig(fld + '_graph.png')
    plt.close()
    return


'''*************************************
    this function produce the ARIMA forecast and error rate
    input:
        1) dataframe that contains the data series
        2) show- boolean if True shows the prediction and actual visually
        3) parameter order: (p,d,q)
            p is the number of autoregressive terms - lagged term
            d is the number of nonseasonal differences needed for stationarity, and ---1=first derivative; 2=second derivatives; 3=third derivatives
            q is the number of lagged forecast errors in the prediction equation.e.g. moving average
        4) training and testing dataset split
    output:
         error of the fitted parameters in ARIMA model - per testing against testing data
'''


def forecasting(dta_df, fld, show, para_order=(1, 1, 1), train_ratio=0.70, output_prefix=''):
    '''****************
        Step 3: Build the Model
        Step 4: Make Prediction and measure accuracy
        ****************


        ##Step 3: Build the Model
        ##prepare training and testing set
        ##the first 70% be training set, the remaining 30% becomes testing set
    '''
    # select the records that are still null after interpolation
    dta_df = dta_df.dropna()
    size = int(len(dta_df)*train_ratio)
    train_df = dta_df[:size]
    test_df = dta_df[size + 1:]

    # p,d,q
    try:
        res = sm.tsa.ARIMA(train_df, order=para_order).fit()
    # when the parameters do not make sense
    except Exception:
        return 99999999999
    predictions = []
    test_outcome = []
    len_test_df = len(test_df)

    # convert the testing set to a list for result comparision
    test_outcome = test_df.values.tolist()

    # Step 4: Make Prediction and measure accuracy
    # working to see how to make prediction on ARIMA --- 1 = 1st record after training set
    output = res.forecast(len_test_df)
    predictions = output[0]

    # make prediction as data series with index same as test_df
    prediction_df = pd.Series(predictions)
    prediction_df.index = test_df.index

    # combine both actual and predction of test data into data
    data = pd.concat([test_df, prediction_df], axis=1)
    data_name = list(data)[0]
    data.columns = ['actual', 'predicted']

    try:
        error = mean_squared_error(test_outcome, predictions)
    except Exception:  # when the error is too large to be stored in the variable
        print("error too large")
        return 99999999999
    print('Test MSE: %.3f' % error)

    if show:
        data.actual.plot(
            color='blue', 
            grid=True,
            label='actual', 
            title=data_name
        )
        data.predicted.plot(color='red', grid=True, label='predicted')
        plt.legend()
        plt.show()
        plt.close()

    if output_prefix != '':
        data.actual.to_csv(output_prefix + 'actual.csv')
        data.predicted.to_csv(output_prefix + 'predicted.csv')
    return error, data, res


'''*************************************MAIN*************************************

Step 1: Download & Import Data
Files are downloaded from the following paths:
Consumption
https://www.eia.gov/opendata/qb.php?category=870&sdid=ELEC.CONS_TOT.COW-CA-98.M
https://www.eia.gov/opendata/qb.php?category=871&sdid=ELEC.CONS_TOT.PEL-CA-98.M
https://www.eia.gov/opendata/qb.php?category=872&sdid=ELEC.CONS_TOT.PC-CA-98.M
https://www.eia.gov/opendata/qb.php?category=873&sdid=ELEC.CONS_TOT.NG-CA-98.M

Cost:
https://www.eia.gov/opendata/qb.php?category=41619&sdid=ELEC.COST.COW-CA-98.M
https://www.eia.gov/opendata/qb.php?category=41623&sdid=ELEC.COST.PEL-CA-98.M
https://www.eia.gov/opendata/qb.php?category=41624&sdid=ELEC.COST.PC-CA-98.M
https://www.eia.gov/opendata/qb.php?category=41625&sdid=ELEC.COST.NG-CA-98.M

Revenue
https://www.eia.gov/opendata/qb.php?category=1007&sdid=ELEC.REV.CA-RES.M
https://www.eia.gov/opendata/qb.php?category=1008&sdid=ELEC.REV.CA-COM.M
https://www.eia.gov/opendata/qb.php?category=1009&sdid=ELEC.REV.CA-IND.M
https://www.eia.gov/opendata/qb.php?category=1010&sdid=ELEC.REV.CA-TRA.M
https://www.eia.gov/opendata/qb.php?category=1011&sdid=ELEC.REV.CA-OTH.M

Trying out the parameters
For better style, we can even seperate this part of program to another file
'''


'''****************
Step 2: Pre-processing the data
****************
'''
# default to optimize the ARIMA parameters by trying out all the combinations
parameter_op = False
# prediction runs on a particular field
select_fld = True

tuple_shape = ()
output_prefix = ''

# import all the files and generate a consolidated data file at file_path_out
if select_fld:
    #list_flds = ['avg_cost_ng']
    list_flds = ['consumption_ng', 'avg_cost_ng']
    tuple_shape_list = [(8, 0, 3), (12, 1, 3)]
else:
    list_flds = preprocess_data(f_name, list_flds)


# this stores the optimized parameters of each field
flds_para_dict = {}
t_i = 0
# looping through list of fields
for fld in list_flds:
    dta_df = load_data(file_path_out, fld, '2001-01-01', '2018-01-01')

    if parameter_op:  # if needed parameters optimization, step 5 is required
        '''****************
        Step 3: Build the Model
        Step 4: Make Prediction and measure accuracy
        Step 5: Further improvement of model – Fine tuning the parameters
        ****************
        '''
        start = time.time()
        lowest_MSE = 99999999999
        lowest_order = (0, 0, 0)
        for p_para in range(13):
            for d_para in range(3):
                for q_para in range(4):
                    order = (p_para, d_para, q_para)
                    print(order)
                    error, temp_data, temp_model = forecasting(dta_df, fld, False, order, 0.7, fld)
                    # Step 5: Further improvement of model – Fine tuning the parameters
                    if error < lowest_MSE:
                        lowest_MSE = error
                        lowest_order = order
                        lowest_data = temp_data
                        lowest_model = temp_model
        end = time.time()
        print('for the field ' + fld)
        print("Best para is")
        print(lowest_order)
        print('Test MSE: %.3f' % lowest_MSE)
        total_time = (end-start)
        print('it takes %.3f s to compute' % total_time)
        print_plt(lowest_data, fld)
        flds_para_dict[fld] = lowest_order

    else:  # if no need for paramters optimization, step 5 is skipped
        '''****************
        Step 3: Build the Model and Step 4: Make Prediction and measure accuracy
        Step 4: Make Prediction and measure accuracy
        ****************
        '''
        tuple_shape = tuple_shape_list[t_i]
        if tuple_shape == ():
            tuple_shape = (7, 1, 2)
        error, temp_data, lowest_model = forecasting(dta_df, fld, True, tuple_shape, 0.9, fld)
        print_plt(temp_data, fld)
        t_i  += 1

    # save model
    f_ARIMA = open(fld + '.pkl', "wb+")
    pickle.dump(lowest_model, f_ARIMA)
    f_ARIMA.close()

f_output = open('result.txt', 'w+')
for keys, values in flds_para_dict.items():
    f_output.write(keys + '\t' + str(values) + '\n')
f_output.close()
'''
avg_cost_pc	(5, 1, 2)
revenue_ind	(8, 2, 3)
consumption_ng	(8, 0, 3)
consumption_pc	(6, 1, 3)
revenue_all	(12, 2, 2)
revenue_other	(8, 0, 3)
avg_cost_ng	(12, 1, 3)
revenue_commerical	(3, 1, 2)
revenue_trans	(10, 1, 0)
consumption_coal	(1, 1, 0)
consumption_pl	(11, 0, 2)
avg_cost_pl	(12, 0, 1)
avg_cost_coal	(1, 0, 3)
revenue_residential	(7, 1, 2)
'''
