import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tomato_preprocess import PreProcess
from tomato_visualize import Visualize


"""파일 설정(시작)"""
# 파일이름
date = f"091622"
ver = "test33"
# lr = 'na'
# epochs = 'na'
lr = 0.001
epochs = 100
description = f"FINAL{lr}_{epochs}"
file_name = f"{date}({ver})_{description}"
print(f"file_name = {file_name}")

# 전처리 클래스
data_directory = "./tomato/"
pp = PreProcess(data_directory=data_directory, outputBy_colName='Sample_no')
vs = Visualize(data_directory=data_directory, outputBy_colName='Sample_no')
"""파일 설정(시작)"""


"""데이터 불러오기_인풋 (시작)"""
# vs.show_scatter_controlled_columns_by(pp.merge_in, '일')
# exit()

# print(inputs.dtypes)
# print("-")
# inputs, test_inputs = pp.input_dfs_outlier_handled() #이상치만 처리한 버전
inputs, test_inputs = pp.random_forest_model() # 이상치 처리 이후 재배형태/품종 예측한 버전


# nan 포함된 열 드랍
inputs = inputs.dropna(axis=1)
test_inputs = test_inputs.dropna(axis=1)
print(inputs.columns)
# exit()
# print(inputs.dtypes)
# print('-')
inputs, test_inputs = [pp.fill_in_the_unobserved(df) for df in [inputs, test_inputs]] # 미관측치 업데이트한 버전 #todo-켜기
inputs, test_inputs = [pp.input_data_transformation(df) for df in [inputs, test_inputs]] # 데이터 형식 #todo-켜기
inputs, test_inputs = [pp.input_dfs_feature_engineered(df) for df in [inputs, test_inputs]] # Feature engineering 된 버전 #todo-켜기

# nan 포함된 열 드랍
# inputs = inputs.dropna(axis=1)
# test_inputs = test_inputs.dropna(axis=1)
# exit()

#아웃풋
outputs = pp.output_df_outlier_handled()
output_sample = pp.test_out





# 중복/시계열 제거하기 (sum 버전)
inputs, outputs = pp.integrate_input_df_by_samples(inputs, outputs, method='sum')
print(test_inputs.shape, output_sample.shape)
test_inputs, output_sample = pp.integrate_test_input_df_by_samples(test_inputs, output_sample,method='sum')
print(test_inputs.shape, output_sample.shape)

print(test_inputs.columns.tolist()==inputs.columns.tolist())

# exit()
# 중복/시계열 제거하기 (mean 버전)
# inputs, outputs = pp.integrate_input_df_by_samples(inputs, outputs, method='mean')
# test_inputs, _ = pp.integrate_input_df_by_samples(test_inputs, method='mean')

# pp.print_all_data()
# print(inputs.head())
# print(test_inputs.head())
# exit()


# print(test_inputs.shape)
# print(test_inputs)

#인풋 이후 결과물을 시각화하려면
# merge = pd.concat([inputs, test_inputs], axis=0, ignore_index=True)
# vs.show_scatter_controlled_columns_by(merge, '주차')
# vs.show_scatter_controlled_columns_by(merge, '일(주)', "이상시 처리 후 INPUT")
# print(inputs.dtypes)
# print(inputs['주차'].astype(int))

# exit()
# vs.show_scatter_controlled_columns_by(merge, '고온', "이상시 처리 후 INPUT")
# vs.show_scatter_controlled_columns_by(merge, '고습', "이상시 처리 후 INPUT")
# vs.show_scatter_controlled_columns_by(merge, '일', "이상시 처리 후 INPUT")

# 데이터 불러오기 (output)
# outputs = pp.train_out

# print(outputs.dtypes)
# print(f"OUTPUT PROCESSED : {pp.train_out.shape} >> {outputs.shape}")

vs.show_scatter_outputs(pp.train_out, "TRAIN_OUTPUT BY 주차 (원본)")
vs.show_scatter_outputs(outputs, "TRAIN_OUTPUT BY 주차 (이상치수정)")
"""데이터 불러오기_아웃풋 (끝)"""

##

"""아웃풋 <= 0 드랍 (시작)"""
count_del_row = 0
for col in ["생장길이", "줄기직경", "개화군"]:
    print(f"DROPPING ROWS IN {col}")
    len_negative = outputs.loc[outputs[col]<= 0]['Sample_no'].tolist()
    count_del_row+= len(len_negative)
    for sample_no in len_negative:
        inputs = inputs.drop(inputs.loc[inputs['Sample_no']==sample_no].index)
        outputs = outputs.drop(outputs.loc[outputs['Sample_no']==sample_no].index)
    print(f"{len(len_negative)} ROWS DELETED FROM {col}. [shape: {inputs.shape} >>{outputs.shape}]")
print(f"******************************TOTAL {count_del_row} ROWS DELETED******************************")
# vs.show_scatter_outputs(outputs, "TRAIN_OUTPUT BY 주차 (<=0 드랍이후)")
"""아웃풋 <= 0 드랍 (끝)"""

##
# print(inputs['급액EC단계'].value_counts())
# exit()
#
# inputs.to_csv("input(lsy).csv", index=False)
# test_inputs.to_csv("test_input(lsy).csv", index=False)
# outputs.to_csv("output(lsy).csv")
# output_sample.to_csv('test_output(lsy).csv', index=False)
#
# exit()

# DataType 정리



# 품종 / 급액EC단계 원핫인코딩
print(inputs.dtypes)

# for df in [inputs, test_inputs]:
#     df = pd.get_dummies(df, columns=['품종', '급액EC단계', '주차'])



inputs = inputs.drop(['품종', '재배형태'], axis=1)
test_inputs = test_inputs.drop(['품종', '재배형태'], axis=1)

# print(inputs.columns)
# print(test_inputs.columns)
# exit()

# inputs = pd.get_dummies(inputs, columns=['품종'])
# test_inputs = pd.get_dummies(test_inputs, columns=['품종'])

# exit()
#
# inputs = pd.get_dummies(inputs, columns=['급액EC단계'])
# test_inputs = pd.get_dummies(test_inputs, columns=['급액EC단계'])

# print(inputs.dtypes)
# print(inputs.columns)

# exit()
"""
BASELINE
# nan 제거  -- 베이스라인이므로 간단한 처리를 위해 nan 항목 보간 없이 학습
inputs = inputs.dropna(axis=1)
# 주차 정보 수치 변환
inputs['주차'] = [int(i.replace('주차', "")) for i in inputs['주차']]
"""

# print(inputs.columns)
# print(inputs)
# print(outputs)

# exit()
#

# scaler
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()


# scaling
input_sc = input_scaler.fit_transform(inputs.iloc[:,3:].to_numpy())
output_sc = output_scaler.fit_transform(outputs.iloc[:,3:].to_numpy())

print(f"INPUT/OUTPUT SHAPE AFTER SCALING: {input_sc.shape}, {output_sc.shape}")
# exit()

"""
요약
"""
# print(inputs.isnull().sum())
# print(test_inputs.isnull().sum())
# print(f"input_sc.shape = {input_sc.shape}")
# print(inputs.columns)
# print(test_inputs.columns)
# exit()


"""
# 입력 시계열화
input_ts = []
for i in outputs['Sample_no']:
    sample = input_sc[inputs['Sample_no'] == i]
    # if len(sample) < 7:
    #     print(f"{i} - {len(sample)}")
    #     sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample,
    #                        axis=0)
    #     print(f"{i}STOP***********************************************\n\n\n\n\n")
    sample = np.expand_dims(sample, axis=0)
    input_ts.append(sample)
input_ts = np.concatenate(input_ts, axis=0)

print(input_ts.shape, output_sc.shape)
print(input_ts)
print(output_sc)

exit()
"""
# 셋 분리
train_x, val_x, train_y, val_y = train_test_split(input_sc, output_sc, test_size=0.2,
                                                  shuffle=True, random_state=0)
# train_x, val_x, train_y, val_y = train_test_split(input_ts, output_sc, test_size=0.2,
#                                                   shuffle=True, random_state=0)


# 모델 정의
def creat_Dense_model(_in_feature_count, _out_feature_count):
    """
    단순 Dense 모델
    :param _in_feature_count: input data feature number
    :param _out_feature_count: output data feature number
    :return: model
    """
    _model = Sequential()

    _model.add(Dense(1000, input_dim=_in_feature_count, kernel_initializer='uniform', activation='relu'))
    _model.add(Dense(500, activation='relu'))
    _model.add(Dense(100, activation='relu'))
    _model.add(Dense(50, activation='relu'))
    _model.add(Dense(10, activation='relu'))
    _model.add(Dense(5, activation='relu'))
    _model.add(Dense(_out_feature_count, kernel_initializer='uniform', activation='relu'))
    return _model

model = creat_Dense_model(input_sc.shape[1], output_sc.shape[1])

model.summary()
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=file_name+'.h5',
                               verbose=1, save_best_only=True, save_weights_only=True)
model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['mse'])

"""
# 모델 정의
def create_model():
    x = Input(shape=[7, input_sc.shape[1]])
    l1 = LSTM(64)(x)
    out = Dense(3, activation='tanh')(l1)
    return Model(inputs=x, outputs=out)

model = create_model()
model.summary()
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=file_name+'.h5',
                               verbose=1, save_best_only=True, save_weights_only=True)
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse'])

"""

# 학습
hist = model.fit(train_x, train_y, batch_size=32, epochs=epochs, validation_data=(val_x, val_y), callbacks=[checkpointer])


# loss 히스토리 확인
fig, loss_ax = plt.subplots()
loss_ax.plot(hist.history['loss'], 'r', label='loss')
loss_ax.plot(hist.history['val_loss'], 'g', label='val_loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend()
plt.title('Training loss - Validation loss plot')
plt.show()

check_save = input("DO YOU WANT TO SAVE THIS? [Y / n]")

if check_save.lower() == 'y':


    # 저장된 가중치 불러오기
    model.load_weights(file_name+'.h5')
    print(f"weights loaded from {file_name}+'.h5")

    """
    베이스라인 워본
    # 테스트셋 전처리 및 추론
    # test_inputs = pd.read_csv('./test_input.csv')
    # output_sample = pd.read_csv('./answer_sample.csv')
    # test_inputs = test_inputs[inputs.columns]
    # test_inputs['주차'] = [int(i.replace('주차', "")) for i in test_inputs['주차']]
    """

    test_input_sc = input_scaler.transform(test_inputs.iloc[:,3:].to_numpy())

    # test_input_ts = []
    # for i in output_sample['Sample_no']:
    #     sample = test_input_sc[test_inputs['Sample_no'] == i]
    #     if len(sample) < 7:
    #         sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample,
    #                            axis=0)
    #         print("STOP***********************************************\n\n\n\n\n")
    #
    #     sample = np.expand_dims(sample, axis=0)
    #     test_input_ts.append(sample)
    # test_input_ts = np.concatenate(test_input_ts, axis=0)

    prediction = model.predict(test_input_sc)

    print(f"input_sc.shape = {input_sc.shape}")
    print(f"output_sc.shape = {output_sc.shape}")
    print(f"test_input_sc.shape = {test_input_sc.shape}")
    print(f"output_sample.shape = {output_sample.shape}")

    prediction = output_scaler.inverse_transform(prediction)
    output_sample[['생장길이', '줄기직경', '개화군']] = prediction


    # 제출할 추론 결과 저장
    output_sample.to_csv(file_name+'.csv', index=False)
    print(f"output saved as {file_name}.csv")

else:
    print('EXIT')
    exit()