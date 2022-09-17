import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class PreProcess():
    def __init__(self, data_directory, outputBy_colName):

        # 데이터 불러오기
        self.train_in = pd.read_csv(data_directory + 'train_input.csv')
        self.train_out = pd.read_csv(data_directory + 'train_output.csv')
        self.test_in = pd.read_csv(data_directory + 'test_input.csv')
        self.test_out = pd.read_csv(data_directory + 'answer_sample.csv')

        # train_in / test_in 병합된 데이터 프레임
        self.merge_in = pd.concat([self.train_in, self.test_in], ignore_index=True, axis=0)

        # 이상치 수정한 train in, test in
        self.train_in_otl, self.test_in_otl = self.input_dfs_outlier_handled()

    def print_all_data(self, display_row = True, display_col = True):
        """
        판다스 출력되는 데이터 모두 보여줄 것인지에 대한 설정
        :param display_row: row만 다 보려면 True or False
        :param display_col: column만 다 보려면 True or False
        :return:
        """
        if display_row:
            pd.set_option('display.max_rows', None)
        if display_col:
            pd.set_option('display.max_columns', None)

        if (display_row is True) and (display_col is True):
            print(f"=== all data will be printed ===")

    def transfrom_data(self, df):
        """
        데이터 변환함수
        :param df:  변환할 데이터 프레임
        :return: 일 >> 데이터, 주차>> '주차' 제거, 시설 >> 'farm'제거
        """
        df['일'] = pd.to_datetime(df['일'].astype(str)).dt.date
        df['주차'] = df['주차'].str.replace("주차", "").astype(int)
        df['시설ID'] = df['시설ID'].str.replace("farm", "").astype(int)

        return df

    def expand_time_column(self, df, col_from):
        """
        int 형태의 날짜 데이터를 가지고 연, 월, 주, 일 변환 및 기존 일을 datetime 형식으로 변환
        :param df: 시간을 추가할 데이터 프레임
        :param col_from: 날짜데이터를 가지고있는 칼럼 이름 (str 형식)
        :return: 연, 월, 일, 주, 일 칼럼 추가 및 기존 일칼럼을 데이터형식으로 변환
        """
        df[col_from] = pd.to_datetime(df[col_from].astype(str))
        df['일(연)'] = df['일'].dt.year
        df['일(월)'] = df['일'].dt.month
        df['일(주)'] = df['일'].dt.week
        df['일(일)'] = df['일'].dt.day
        df['일'] = df['일'].dt.date

        return df

    def correct_outliers_with_mean(self, df, df_outliers, col_outlier, col_groupBy):
        """
        이상치를 주어진 조건의 평균값으로 대체하는 함수
        :param df: 이상치가 포함되어있어 수정되어야 할 데이터프레임
        :param df_outliers: 이상치로만 구성된 데이터 프레임
        :param col_outlier: 수정해야할 칼럼
        :param col_groupBy: 평균을 적용하기 위한 시간기준 (연/월/주/일)
        :return: 이상치가 주어진 기준의 평균값으로 업데이트 된 데이터 프레임
        """
        # 주어진 조건 기준으로 데이터를 group-by
        grp = df.groupby([col_groupBy], as_index=False).mean()[[col_groupBy, col_outlier]]

        # 이상치 처리에 필요한 데이터
        count_outlier = len(df_outliers) #이상치의 갯수
        idx_outliers = df_outliers.index.tolist() #이상치가 포함된 데이터프레임의 인덱스
        value = df_outliers[col_groupBy].tolist() #이상치가 있는 데이터프레임에서 그룹바이할 칼럼의 값들을 불러옴
        new_values = [grp.loc[grp[col_groupBy] == v][col_outlier].iloc[0] for v in value] #새로 업데이트 할 값

        for i in range(count_outlier):
            index = idx_outliers[i] # 데이터 프레임 내 이상치의 인덱스
            new_val = new_values[i] # 업데이트할 값
            df.at[index, col_outlier] = new_val #데이터프레임에 업데이트

        return df

    def correct_outliers_with_multiplication(self, df, df_outliers, col_outlier, multiply_by):
        """
        이상치를 이상치*주어진 값으로 대체하는 함수
        :param df: 이상치가 포함되어있어 수정되어야 할 데이터프레임
        :param df_outliers: df_outliers: 이상치로만 구성된 데이터 프레임
        :param col_outlier: 수정해야할 칼럼
        :param multiply_by: 이상치에 곱해줄 값
        :return: 이상치가 이상치*곱으로 업데이트 된 데이터 프레임
        """

        # STEP 1: 이상치 있는 행들의 인덱스 번호 수집
        idx_outliers = df_outliers.index.tolist()

        # STEP 2: 인덱스로 해당 데이터를 df내에서 접근. old_value 조회 > new_value 계산 > row 내 값 수정
        for idx in idx_outliers:
            old_val = df.iloc[idx][col_outlier] #업데이트가 필요한 이상치의 값
            new_val = old_val * multiply_by # 이상치에 적용할 새로운 값
            df.at[idx, col_outlier] = new_val # 새 값으로 업데이트

        return df

    def input_dfs_outlier_handled(self):
        """
        ['내부CO2', '내부습도', '내부온도', '일사량'] 열 내 이상치를 수정
        :return: 이상치가 수정된 데이터 프레임이 다음 형식으로 출력: [train_in, test_in]
        """
        df_inputs = [self.train_in, self.test_in]
        result = [] #이상치가 수정 이후 데이터프레임들을 저장할 리스트

        for df in df_inputs:
            # STEP 0: 일 >> ['일(연)', '일(월)', '일(주)', '일(일)']
            df = self.expand_time_column(df, '일')

            # STEP 1: 이상치 제거
            # STEP 1-1. 내부CO2
            df_outliers = df.loc[(df['내부CO2'] > 900) | (df['내부CO2'] < 200)]
            df = self.correct_outliers_with_mean(df= df, df_outliers=df_outliers, col_outlier='내부CO2', col_groupBy='일(월)')
            # print("==내부CO2 결측치 처리완료==")
            # print(df.loc[(df['내부CO2'] > 900) | (df['내부CO2'] < 200)])

            # STEP 1-2. 내부습도
            df_outliers = df.loc[(df['내부습도'] > 100)]
            df = self.correct_outliers_with_multiplication(df= df, df_outliers=df_outliers, col_outlier='내부습도', multiply_by=0.1)
            # print("==내부습도 이상치 (1/2) 처리완료==")
            # print(df.loc[(df['내부습도'] > 100)])

            # STEP 1-2-2. 내부습도 20인 행은 월평균으로
            df_outliers = df.loc[(df['내부습도'] < 20)]
            df = self.correct_outliers_with_mean(df= df, df_outliers= df_outliers, col_outlier='내부습도', col_groupBy='일(월)')
            # print("==내부습도 이상치 (2/2) 처리완료==")
            # print(df.loc[(df['내부습도'] < 20)])

            # STEP 1-3. 내부온도 7미만인 행은 월평균으로
            df_outliers = df.loc[(df['내부온도'] < 7)]
            df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='내부온도', col_groupBy='일(월)')
            # print("==내부온도 이상치 처리완료==")
            # print(df.loc[(df['내부온도'] < 7)])

            # STEP 1-4. 일사량 0인 행은 월평균으로
            df_outliers = df.loc[df['일사량'] == 0]
            df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='일사량', col_groupBy='일(월)')
            # print("==일사량 이상치 처리완료==")
            # print(df.loc[df['일사량'] == 0])

            result.append(df)

            print("OUTLIERS CORRECTED")

        return result

    def output_df_outlier_handled(self):
        """
        트레인 아웃풋 내 이상치 수정
        :return: 이상치가 수정된 트레인 아웃풋
        """
        df = self.train_out
        df['주차'] = df['주차'].str.replace("주차", "").astype(int)

        #STEP 1: 이상치 제거
        # STEP 1-1. 생장길이 (600초과하는 값은 10으로 나눠줄 것)
        df_outliers = df.loc[(df['생장길이'] > 600)]
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='생장길이', multiply_by=0.1)
        # print("==생장길이 이상치 처리완료==")
        # print(df.loc[(df['생장길이'] > 600)])

        # STEP 1-2. 생장길이 (400초과하는 값은 각 주차의 평균으로 수정)
        df_outliers = df.loc[(df['생장길이'] > 400) | (df['생장길이'] <= 0)]
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='생장길이', col_groupBy='주차')
        # print("==생장길이 이상치 처리완료==")
        # print(df.loc[(df['생장길이'] > 400) | (df['생장길이'] <= 0)])

        # STEP 2-1. 줄기직경 (60초과하는 값은 10으로 나눠줄 것)
        df_outliers = df.loc[df['줄기직경'] > 60]
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='줄기직경', multiply_by=0.1)
        # print("==줄기직경 이상치 처리완료==")
        # print(df.loc[df['줄기직경'] > 60])

        # STEP 2-2. 줄기직경 (15 초과 또는 0 이하의 값은 각 주차의 평균으로 수정)
        df_outliers = df.loc[(df['줄기직경'] >15 ) | (df['줄기직경'] <= 0)]
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='줄기직경', col_groupBy='주차')
        # print("==줄기직경 이상치 처리완료==")
        # print(df.loc[(df['줄기직경'] >15 ) | (df['줄기직경'] < 2.5)])

        # STEP 3-1. 개화군 (개화군 40 초과할 경우 10으로 나눠줄 것)
        df_outliers = df.loc[df['개화군'] > 40]
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='개화군', multiply_by=0.1)
        # print("==개화군 이상치 처리완료==")
        # print(df.loc[df['개화군'] > 40])

        # STEP 3-2. 개화군 (개화군 25 초과 또는 0 이하일 경우 각 주차의 평균으로 대체)
        df_outliers = df.loc[(df['개화군'] > 25) | (df['개화군'] <= 0)]
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='개화군', col_groupBy='주차')
        # print("==개화군 이상치 처리완료==")
        # print(df.loc[(df['개화군'] > 25) | (df['개화군'] <= 0)])

        return df

    def random_forest_model(self, show_chart=False):
        """
        이상치 수정된 데이터를 기반으로 train / test input 내 재배형태/품종을 RandomForestClassifier로 채움
        :param show_chart: 첫번째 Decision Tree의 차트를 보려면 True
        :return: Random Forest Classifier로 재배형태/품종 결측치를 채운 데이터프레임
        """
        train_in = self.train_in_otl
        test_in = self.test_in_otl

        # Merge 이후 식별/분리 쉽게 구분열 추가
        train_in['구분'] = 'train'
        test_in['구분'] = 'test'

        # 데이터 통합 및 변환
        df_merge = pd.concat([train_in, test_in], ignore_index=True, axis=0)
        df = self.transfrom_data(df_merge)

        # 분류기의 인풋/아웃풋 열
        cols_input = ['내부CO2', '내부습도', '내부온도', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)']
        cols_output = ['품종', '재배형태']


        cols_io = cols_input+cols_output
        data_types = df[cols_io].dtypes

        for col_output in cols_output:
            # 결측치 채울 칼럼 내 데이터 변환 (str >> num)
            unique = df.loc[df[col_output].notnull(), col_output].unique()  # 결측치 채울 칼럼의 values
            for i, val in enumerate(unique):
                i += 1  # 인덱스를 1부터 시작하기 위한 장치. 고유값의 번호를 1부터 매기기시작하기 위함. ex) 수경 1, 토경 2
                df.loc[(df[col_output].notnull())&(df[col_output] == val), col_output] = int(i)

            # 전체 데이터 변환
            for col in cols_input:
                # Float 타입의 데이터는 flaot로
                if data_types[col] == 'float64' or data_types[col] == 'float32':
                    df[col] = df.loc[df[col].notnull(), col].astype(float)
                # 그 외의 경우 int로
                else:
                    df[col] = df.loc[df[col].notnull(), col].astype(int)

            # 이전 분류기 결과를 인트로
            df[col_output] = df.loc[df[col_output].notnull()][col_output].astype(int)

            # 각 샘플들의 평균
            group_sample = df.groupby(['Sample_no'], as_index=False).mean()

            df_train = group_sample.loc[group_sample[col_output].notnull()]  # 샘플별 각 항목 평균 + col_output null 포함하지 않는 칼럼
            df_test = group_sample.loc[group_sample[col_output].isnull()]  # 샘플별 각 항목 평균 + col_output null 포함하는 칼럼

            # 분류기에 넣을 x, y값들
            train_x = df_train[cols_input].to_numpy()
            train_y = df_train[col_output].to_numpy()
            test_x = df_test[cols_input].to_numpy()

            clf = RandomForestClassifier(n_estimators=100, max_depth=len(cols_input), random_state=0)
            clf.fit(train_x, train_y)
            test_y = clf.predict(test_x) # 샘플별 예측값

            df_test[col_output] = test_y #예측된 값을 데이터프레임에 채워넣음
            for sample_no in df_test['Sample_no'].unique():
                predict = df_test.loc[df_test['Sample_no'] == sample_no][col_output].iloc[0]
                df.loc[df['Sample_no'] == sample_no, col_output] = predict

            # 첫번째 트리의 그래프
            if show_chart is True:
                plot_tree(clf.estimators_[0], feature_names=cols_input, class_names=unique,fontsize=8)
                plt.title(f"Estimator[0] - {col_output}")
                plt.show()

        # 통합된 데이터를 train_in, test_in으로 분할 및 구분칼럼 삭제
        train_in = df.loc[df['구분'] == 'train'].drop(columns = ['구분'], axis=1)
        test_in = df.loc[df['구분'] == 'test'].drop(columns = ['구분'], axis=1)

        print("RANDOM_FOREST_CLASSIFIER COMPLETED")

        return train_in, test_in

    def fill_in_the_unobserved(self, df_input):
        """
        관측되지 않은 날짜를 추론하고 전날 미관측치 전날 데이터로 채워넣음
        :param df_input: 미관측치가 있는 데이터 프레임
        :return: 미관측치가 채워진 데이터 프레임
        """

        # 관측치 7개 미만인 샘플들의 df
        df_less7 = df_input[df_input['Sample_no'].map(df_input['Sample_no'].value_counts() < 7)]

        # 새로 생성될 행을 임시로 저장할 df
        df_newRows = pd.DataFrame(data=None, columns=df_input.columns)
        sample_less7 = df_less7['Sample_no'].unique() #관측이 7회 미만 이루어진 샘플들의 번호 리스트

        for sample_no in sample_less7:
            df_sample = df_input[df_input['Sample_no'] == sample_no] # 각 샘플의 관측치

            # 미관측치가 있는 날짜 추론
            actual_dates = df_sample['일'].to_list() #자료 내에서 관측이 이루어진 날짜
            first_date = df_sample['일'].min() # 첫 관측이 이루어진 날짜
            tobe_dates = [first_date + timedelta(days=i) for i in range(0, 7)] # 첫 관측이 이루어진 날 기점으로 7일까지의 날짜 리스트
            """
            tobe-dates
            대전제: 첫 관측이 이루어지는 날짜를 알 수 없기 때문에 데이터에서 처음 관측된 날을 1주일 관측의 첫날로 간주
            """
            # 미관측치 날짜 도출
            missing_dates = set(tobe_dates).difference(set(actual_dates)) # 차집합을 이용해서 관측되지 않은 날짜 도출
            missing_dates = sorted(missing_dates) # 미관측 날짜를 순서대로 정렬

            # missing date 하나당 채워넣기 작업 알고리즘
            for date in missing_dates:
                # date = 미관측 된 날짜
                date_yesterday = date - timedelta(days=1) #미관측치 전날
                data_yesterday = df_sample[df_sample['일'] == date_yesterday] #전날의 데이터 불러오기

                if len(data_yesterday) == 0: # 전날 데이터가 없을 경우, 샘플의 가장 처음으로 관측된 데이터를 전날의 데이터로 간주
                    data_yesterday = df_sample.head(1)

                new_data = data_yesterday
                new_data.loc[new_data['일'] == date_yesterday, ['일']] = date

                df_newRows = pd.concat([df_newRows, new_data], axis=0, ignore_index=True) #
                df_sample = pd.concat([df_sample, new_data], axis=0, ignore_index=True)

        # 새로운 열들을 기존 데이터프레임에 추가

        df_input = pd.concat([df_input, df_newRows], axis=0, ignore_index=True)
        # 통합된 데이터를 날짜 (오름차순), 샘플번호(오름차순)으로 정렬
        df_input = df_input.sort_values(by=['일'])
        df_input = df_input.sort_values(by=['Sample_no'])
        df_input = df_input.reset_index(drop=True) #인덱스 번호를 초기화

        print("FILL-IN COMPLETED")

        return df_input

    def input_data_transformation(self, df):

        col_trfm =['Sample_no', '시설ID', '주차', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)', '급액횟수', '급액량(회당)']
        col_str2int = ['Sample_no', '시설ID', '주차', '품종', '재배형태', '급액횟수', '급액량(회당)']
        col_date2int = ['일(연)', '일(월)', '일(주)', '일(일)']
        """
        배운점: 데이터 변환은 데이터 분석 가장 첫단계에 이루어져야함. 
        중간중간 변환하다보니 데이터타입의 일관성이 의도치않게 중간에 훼손되는 경우가 많음
        """

        for col in col_trfm:
            # print(f"{col} - {df[col].dtypes}")
            if col in col_str2int:
                df[col] = df[col].astype(int)
            elif col in col_date2int:
                df[col] = df[col].astype(str)
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].astype(float)
        return df

    def input_dfs_feature_engineered(self, df):
        """
        기존 데이터프레임을 토대로 특징을 추가
        :param df: 새롭게 도출된 특징을 추가할 데이터 프레임
        :return: 새롭게 도출된 특징이 추가할 데이터 프레임
        """

        # 당일 이루어진 총 급여량
        df['총급여량'] = df['급액량(회당)'] * df['급액횟수']
        # print(df['총급여량'].value_counts())

        # 토마토의 나이를 6개월로 가정하여 나이값 도출
        df['나이'] = df['주차']/24 #6개월(4주*6)
        # print(df['나이'].value_counts())

        # 관측된 날짜의 온도가 개화에 적정한 범위에 있었는지를 이분법으로 표현
        df['개화적온'] = 0
        df.loc[(df['내부온도']>=20) & (df['내부온도']<=25), '개화적온'] = 1
        # print(df['개화적온'].value_counts())

        # 관측된 날짜의 습도가 토마토에 적정한 범위에 있었는지를 이분법으로 표현
        df['적정습도'] = 0
        df.loc[(df['내부습도']>=65) & (df['내부습도']<=80), '적정습도'] = 1
        # print(df['적정습도'].value_counts())

        # 관측된 날짜의 CO2가 토마토에 적정한 범위에 있었는지를 이분법으로 표현
        df['적정CO2'] = 0
        df.loc[(df['내부CO2']>=400) & (df['내부습도']<=700), '적정CO2'] = 1
        # print(df['적정CO2'].value_counts())

        # 관측된 날짜의 CO2가 토마토에 적정한 범위에 있었는지를 이분법으로 표현
        df['적정온도'] = 0
        df.loc[(df['내부온도']>=20) & (df['내부온도']<=25), '적정온도'] = 1
        # print(df['적정온도'].value_counts())

        # 관측된 날짜의 습도가 과했는지를 이분법으로 표현
        df['고습'] = 0
        df.loc[(df['내부습도']>85), '고습'] = 1
        # print(df['고습'].value_counts())

        # 관측된 날짜의 온도가 과했는지를 이분법으로 표현
        df['고온'] = 0
        df.loc[(df['고온']>28), '고온'] = 1
        # print(df['고온'].value_counts())

        # 관측된 날짜의 습도가 너무 낮았는지를 이분법으로 표현
        df['건조'] = 0
        df.loc[(df['내부습도']<40), '건조'] = 1
        # print(df['건조'].value_counts())

        # 관측된 날짜의 온도가 너무 낮았는지를 이분법으로 표현
        df['냉해'] = 0
        df.loc[(df['내부온도']<10), '냉해'] = 1
        # print(df['냉해'].value_counts())

        # df['급액EC단계'] = 0 #모두 3단계였으므로 사용하지않음
        # df.loc[(df['급액EC(dS/m)']>0) & (df['급액EC(dS/m)']<=0.8), '급액EC단계'] = 1
        # df.loc[(df['급액EC(dS/m)']>0.8) & (df['급액EC(dS/m)']<=1.5), '급액EC단계'] = 2
        # df.loc[(df['급액EC(dS/m)']>1.5), '급액EC단계'] = 3
        # print(df['급액EC단계'].value_counts())

        # 관측된 날짜의 분기에 가중치를 부여함
        df['분기'] = 0
        df.loc[(df['일(월)']>=0) & (df['급액EC(dS/m)']<4), '분기'] = 3
        df.loc[(df['일(월)']>=4) & (df['급액EC(dS/m)']<7), '분기'] = 2
        df.loc[(df['일(월)']>=7) & (df['급액EC(dS/m)']<10), '분기'] = 1
        df.loc[(df['일(월)']>=10) & (df['급액EC(dS/m)']<13), '분기'] = 4
        # print(df['분기'].value_counts())

        return df

    def one_hot_encoding(self, df, list_columns):
        df = pd.get_dummies(df, columns = list_columns)

        return df


