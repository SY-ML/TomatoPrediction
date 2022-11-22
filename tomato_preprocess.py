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

        # train_in / test_in 병합
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
        Switches outliers with a desired mean value
        :param df: a dataframe containing outliers
        :param df_outliers: a dataframe only with outliers
        :param col_outlier: the column in which outliers are to be corrected
        :param col_groupBy: the column that the mean is calculated from to correct outliers
        :return: dataframe after correcting outliers
        """
        grp = df.groupby([col_groupBy], as_index=False).mean()[[col_groupBy, col_outlier]]
        count_outlier = len(df_outliers)
        # print(f"count_outlier = {count_outlier}")

        idx_outliers = df_outliers.index.tolist()
        # print(f"idx_outliers = {idx_outliers}")
        month = df_outliers[col_groupBy].tolist()
        # print(f"month = {month}")
        new_values = [grp.loc[grp[col_groupBy] == m][col_outlier].iloc[0] for m in month]
        # print(f"new_value = {new_values}")

        for i in range(count_outlier):
            index = idx_outliers[i]
            new_val = new_values[i]
            # print(f"new_values[{i}] = {new_val}")
            # print(f"index = {index}")
            # print("before")
            # print(df.iloc[index])
            df.at[index, col_outlier] = new_val
            # print("after")
            # print(df.iloc[index])

        return df
    #
    def correct_outliers_with_multiplication(self, df, df_outliers, col_outlier, multiply_by):
        """
        Switches outliers with a desired mean value
        :param df: a dataframe containing outliers
        :param df_outliers: a dataframe only with outliers
        :param col_outlier: the column in which outliers are to be corrected
        :param multiply_by: a value to multiply outliers by
        :return: dataframe after correcting outliers
        """
        # print(df_outliers)
        # STEP 1: 이상치 있는 행들의 인덱스 번호 수집
        idx_outliers = df_outliers.index.tolist()
        # STEP 2: 인덱스로 해당 데이터를 df내에서 접근. old_value 조회 > new_value 계산 > row 내 값 수정
        for idx in idx_outliers:
            # print(f"idx = {idx}")
            old_val = df.iloc[idx][col_outlier]
            # print(old_val)
            new_val = old_val * multiply_by
            # print(new_val)
            # print(df.iloc[idx])
            # print("-")
            df.at[idx, col_outlier] = new_val
            # print(df.iloc[idx])
            # print("==")
        return df

    def input_dfs_outlier_handled(self):
        """
        ['내부CO2', '내부습도', '내부온도', '일사량'] 열의 이상치를 수정
        :return: 이상치가 수정된 데이터 프레임이 다음 형식으로 출력: [train_in, test_in]
        """
        df_inputs = [self.train_in, self.test_in]
        result = []
        for df in df_inputs:
            # print(df['일'].head())
            # print("&&&")
            # STEP 0: 일 >> ['일(연)', '일(월)', '일(주)', '일(일)']
            df = self.expand_time_column(df, '일')
            cols_time = ['일(연)', '일(월)', '일(주)', '일(일)']
            cols_outlier = ['내부CO2', '내부습도', '내부온도', '일사량']
            # print(df.columns)
            # STEP 1: 이상치 제거

            # STEP 1-1. 내부CO2
            # print("==내부CO2 이상치==")
            df_outliers = df.loc[(df['내부CO2'] > 900) | (df['내부CO2'] < 200)]
            # print(df_outliers)
            df = self.correct_outliers_with_mean(df= df, df_outliers=df_outliers, col_outlier='내부CO2', col_groupBy='일(월)')
            # print("==내부CO2 결측치 처리완료==")
            # print(df.loc[(df['내부CO2'] > 900) | (df['내부CO2'] < 200)])


            # STEP 1-2. 내부습도
            # print("==내부습도 이상치 (1/2)==")
            df_outliers = df.loc[(df['내부습도'] > 100)]
            # print(df_outliers)
            df = self.correct_outliers_with_multiplication(df= df, df_outliers=df_outliers, col_outlier='내부습도', multiply_by=0.1)

            # print("==내부습도 이상치 (1/2) 처리완료==")
            # print(df.loc[(df['내부습도'] > 100)])

            # STEP 1-2-2. 내부습도 0인 행은 월평균으로 (코드가 복잡해지므로 월평균에서 0을 제외하지는 않았음)
            # print("==내부습도 이상치 (2/2)==")
            df_outliers = df.loc[(df['내부습도'] < 20)]
            # print(df_outliers)
            df = self.correct_outliers_with_mean(df= df, df_outliers= df_outliers, col_outlier='내부습도', col_groupBy='일(월)')
            # print("==내부습도 이상치 (2/2) 처리완료==")
            # print(df.loc[(df['내부습도'] < 20)])

            # STEP 1-3. 내부온도 10미만인 행은 월평균으로
            # print("==내부온도 이상치==")
            df_outliers = df.loc[(df['내부온도'] < 7)]
            # print(df_outliers)
            df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='내부온도', col_groupBy='일(월)')
            # print("==내부온도 이상치 처리완료==")
            # print(df.loc[(df['내부온도'] < 7)])

            # STEP 1-4. 일사량 0인 행은 월평균으로
            df_outliers = df.loc[df['일사량'] == 0]
            # print(df_outliers)
            df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='일사량', col_groupBy='일(월)')
            # print("==일사량 이상치 처리완료==")
            # print(df.loc[df['일사량'] == 0])

            result.append(df)

            print("****************************************\n\nDATA PROCESSED\n\n****************************************")

        return result

    def output_df_outlier_handled(self):
        """
        아웃풋 이상치 수정
        :return:
        """
        df = self.train_out
        # print(df.columns)
        # print("output")
        # print(df['조사일'].head())
        df['주차'] = df['주차'].str.replace("주차", "").astype(int)

        #STEP 1: 이상치 제거
        # STEP 1-1. 생장길이 (50주차 미만)
        df_outliers = df.loc[(df['생장길이'] > 600)]
        # print("==생장길이 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='생장길이', multiply_by=0.1)
        # print("==생장길이 이상치 처리완료==")
        # print(df.loc[(df['생장길이'] > 600)])

        df_outliers = df.loc[(df['생장길이'] > 400) | (df['생장길이'] <= 0)]
        # print("==생장길이 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='생장길이', col_groupBy='주차')
        # print("==생장길이 이상치 처리완료==")
        # print(df.loc[(df['생장길이'] > 400) | (df['생장길이'] <= 0)])

        # STEP 2. 줄기직경
        df_outliers = df.loc[df['줄기직경'] > 60]
        # print("==줄기직경 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='줄기직경', multiply_by=0.1)
        # print("==줄기직경 이상치 처리완료==")
        # print(df.loc[df['줄기직경'] > 60])

        # STEP 2. 줄기직경
        df_outliers = df.loc[(df['줄기직경'] >15 ) | (df['줄기직경'] <= 0)]
        # print("==줄기직경 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='줄기직경', col_groupBy='주차')
        # print("==줄기직경 이상치 처리완료==")
        # print(df.loc[(df['줄기직경'] >15 ) | (df['줄기직경'] < 2.5)])


        # STEP 3. 개화군
        df_outliers = df.loc[df['개화군'] > 40]
        # print("==개화군 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_multiplication(df = df, df_outliers= df_outliers, col_outlier='개화군', multiply_by=0.1)
        # print("==개화군 이상치 처리완료==")
        # print(df.loc[df['개화군'] > 40])

        df_outliers = df.loc[(df['개화군'] > 25) | (df['개화군'] <= 0)]
        # print("==개화군 이상치==")
        # print(df_outliers)
        df = self.correct_outliers_with_mean(df = df, df_outliers= df_outliers, col_outlier='개화군', col_groupBy='주차')
        # print("==개화군 이상치 처리완료==")
        # print(df.loc[(df['개화군'] > 25) | (df['개화군'] <= 0)])

        return df


    def random_forest_model(self, show_chart=False):
        """
        이상치 수정된 데이터를 기반으로 train / test input 내 재배형태/품종을 RandomForestClassifier로 채움
        :param show_chart: 첫번째 Decision Tree의 차트를 보려면 True
        :return:
        """
        train_in = self.train_in_otl
        test_in = self.test_in_otl

        # Merge 이후 식별/분리 쉽게 구분열 추가
        train_in['구분'] = 'train'
        test_in['구분'] = 'test'

        df_merge = pd.concat([train_in, test_in], ignore_index=True, axis=0)
        df = self.transfrom_data(df_merge)


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
                if data_types[col] == 'float64' or data_types[col] == 'float32':
                    df[col] = df.loc[df[col].notnull(), col].astype(float)
                else:
                    df[col] = df.loc[df[col].notnull(), col].astype(int)

            df[col_output] = df.loc[df[col_output].notnull()][col_output].astype(int)

            group_sample = df.groupby(['Sample_no'], as_index=False).mean()

            df_train = group_sample.loc[group_sample[col_output].notnull()]  # 샘플별 각 항목 평균 + col_output null 포함하지 않는 칼럼
            df_test = group_sample.loc[group_sample[col_output].isnull()]  # 샘플별 각 항목 평균 + col_output null 포함하는 칼럼

            train_x = df_train[cols_input].to_numpy()
            train_y = df_train[col_output].to_numpy()
            test_x = df_test[cols_input].to_numpy()

            clf = RandomForestClassifier(n_estimators=100, max_depth=len(cols_input), random_state=0)
            clf.fit(train_x, train_y)
            test_y = clf.predict(test_x) # 샘플별 예측값

            df_test[col_output] = test_y
            for sample_no in df_test['Sample_no'].unique():
                predict = df_test.loc[df_test['Sample_no'] == sample_no][col_output].iloc[0]
                df.loc[df['Sample_no'] == sample_no, col_output] = predict

            # print("HERE", df[col_output].value_counts(dropna=False))

            if show_chart is True:
                plot_tree(clf.estimators_[0], feature_names=cols_input, class_names=unique,fontsize=8)
                plt.title(f"Estimator[0] - {col_output}")
                plt.show()


        train_in = df.loc[df['구분'] == 'train'].drop(columns = ['구분'], axis=1)
        test_in = df.loc[df['구분'] == 'test'].drop(columns = ['구분'], axis=1)

        print("RANDOM_FOREST_CLASSIFIER : DONE ")

        return train_in, test_in
    #
    # def random_forest_model(self, show_chart=False):
    #     """
    #     이상치 수정된 데이터를 기반으로 train / test input 내 재배형태/품종을 RandomForestClassifier로 채움
    #     :param show_chart: 첫번째 Decision Tree의 차트를 보려면 True
    #     :return:
    #     """
    #     train_in = self.train_in_otl
    #     test_in = self.test_in_otl
    #
    #     # Merge 이후 식별/분리 쉽게 구분열 추가
    #     train_in['구분'] = 'train'
    #     test_in['구분'] = 'test'
    #
    #     df_merge = pd.concat([train_in, test_in], ignore_index=True, axis=0)
    #     df = self.transfrom_data(df_merge)
    #
    #
    #     cols_input = ['내부CO2', '내부습도', '내부온도', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)']
    #     cols_output = ['품종', '재배형태']
    #
    #     cols_io = cols_input+cols_output
    #     data_types = df[cols_io].dtypes
    #
    #     for col_output in cols_output:
    #         # 결측치 채울 칼럼 내 데이터 변환 (str >> num)
    #         unique = df.loc[df[col_output].notnull(), col_output].unique()  # 결측치 채울 칼럼의 values
    #         for i, val in enumerate(unique):
    #             i += 1  # 인덱스를 1부터 시작하기 위한 장치. 고유값의 번호를 1부터 매기기시작하기 위함. ex) 수경 1, 토경 2
    #             df.loc[(df[col_output].notnull())&(df[col_output] == val), col_output] = int(i)
    #
    #         # 전체 데이터 변환
    #         for col in cols_input:
    #             if data_types[col] == 'float64' or data_types[col] == 'float32':
    #                 df[col] = df.loc[df[col].notnull(), col].astype(float)
    #             else:
    #                 df[col] = df.loc[df[col].notnull(), col].astype(int)
    #
    #         df[col_output] = df.loc[df[col_output].notnull()][col_output].astype(int)
    #
    #         group_sample = df.groupby(['Sample_no'], as_index=False).mean()
    #
    #         df_train = group_sample.loc[group_sample[col_output].notnull()]  # 샘플별 각 항목 평균 + col_output null 포함하지 않는 칼럼
    #         df_test = group_sample.loc[group_sample[col_output].isnull()]  # 샘플별 각 항목 평균 + col_output null 포함하는 칼럼
    #
    #         train_x = df_train[cols_input].to_numpy()
    #         train_y = df_train[col_output].to_numpy()
    #         test_x = df_test[cols_input].to_numpy()
    #
    #         clf = RandomForestClassifier(n_estimators=100, max_depth=len(cols_input), random_state=0)
    #         clf.fit(train_x, train_y)
    #         test_y = clf.predict(test_x) # 샘플별 예측값
    #
    #         df_test[col_output] = test_y
    #         for sample_no in df_test['Sample_no'].unique():
    #             predict = df_test.loc[df_test['Sample_no'] == sample_no][col_output].iloc[0]
    #             df.loc[df['Sample_no'] == sample_no, col_output] = predict
    #
    #         # print("HERE", df[col_output].value_counts(dropna=False))
    #
    #         if show_chart is True:
    #             plot_tree(clf.estimators_[0], feature_names=cols_input, class_names=unique,fontsize=8)
    #             plt.title(f"Estimator[0] - {col_output}")
    #             plt.show()
    #
    #
    #     train_in = df.loc[df['구분'] == 'train'].drop(columns = ['구분'], axis=1)
    #     test_in = df.loc[df['구분'] == 'test'].drop(columns = ['구분'], axis=1)
    #
    #     print("RANDOM_FOREST_CLASSIFIER : DONE ")
    #
    #     return train_in, test_in

    def fill_in_the_unobserved(self, df_input):
        """
        미관측치를 전날 데이터에서 복사하여 채워넣음
        :param df_input:
        :return:
        """


        # 관측치 7개 미만인 샘플들의 df
        df_less7 = df_input[df_input['Sample_no'].map(df_input['Sample_no'].value_counts() < 7)]

        # 새로 생성될 행이 추가될 df
        df_newRows = pd.DataFrame(data=None, columns=df_input.columns)

        sample_less7 = df_less7['Sample_no'].unique()
        for sample_no in sample_less7:
            df_sample = df_input[df_input['Sample_no'] == sample_no]

            # 추가해야 할 행의 수
            actual_dates = df_sample['일'].to_list()
            first_date = df_sample['일'].min()
            tobe_dates = [first_date + timedelta(days=i) for i in range(0, 7)]
            missing_dates = set(tobe_dates).difference(set(actual_dates))
            missing_dates = sorted(missing_dates)
            """
            프린트
            # print(f"number of rows to add : {num_needed}")
            # print(f"actual_dates = {actual_dates}")
            # print(f"first/last date in list= {first_date}/{last_date}")
            # print(f"Sample{sample_no}: missing_date(s) = {missing_dates}")
            print(f"tobe_date(s)= {tobe_dates}")
            """
            # print(missing_dates)

            # missing date 하나당 채워넣기 작업 알고리즘
            for date in missing_dates:
                # self.print_all_data(display_row=False)
                # print(df_sample)
                # print(f"sample{sample_no}: new data created on {date}")

                date_yesterday = date - timedelta(days=1)
                data_yesterday = df_sample[df_sample['일'] == date_yesterday]

                if len(data_yesterday) == 0:
                    data_yesterday = df_sample.head(1)
                # print(data_yesterday)

                new_data = data_yesterday
                new_data.loc[new_data['일'] == date_yesterday, ['일']] = date
                # print("=")
                # print(new_data.dtypes)
                # print("=")
                df_newRows = pd.concat([df_newRows, new_data], axis=0, ignore_index=True)
                df_sample = pd.concat([df_sample, new_data], axis=0, ignore_index=True)
                # print(new_data)
                # print("-")
            # print("======\n\n\n\n\n")

        # print(df_newRows)
        # print("-")
        # print(df_input)
        # print(df_newRows.dtypes)

        # print(f"df == df_new {df_input.columns.tolist() == df_newRows.columns.tolist()}")
        # self.print_all_data()
        # print(df_input.head(), df_input.tail())
        # print(df_newRows.head(), df_newRows.tail())
        df_input = pd.concat([df_input, df_newRows], axis=0, ignore_index=True)
        # print("-")
        # print(df_input)
        # 통합된 데이터 솔트
        df_input = df_input.sort_values(by=['일'])
        # print(f"sorted by 일")
        df_input = df_input.sort_values(by=['Sample_no'])
        # print(f"sorted by Sample_no")

        df_input = df_input.reset_index(drop=True)
        # print("index reset completed")
        # print(df_input.columns.tolist())
        # ['Sample_no', '시설ID', '일', '주차', '내부CO2', '내부습도', '내부온도', '일사량', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)']

        print("filled in")

        return df_input

    def input_data_transformation(self, df):
        # print(df.columns.tolist())

        col_trfm =['Sample_no', '시설ID', '주차', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)', '급액횟수', '급액량(회당)']
        col_str2int = ['Sample_no', '시설ID', '주차', '품종', '재배형태', '급액횟수', '급액량(회당)']
        col_date2int = ['일(연)', '일(월)', '일(주)', '일(일)']
        # col_int = ['Sample_no', '시설ID', '일', '주차', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)']
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

        # ['내부CO2', '내부습도', '내부온도', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)']
        df['총급여량'] = df['급액량(회당)'] * df['급액횟수']
        # print(df['총급여량'].value_counts())

        df['나이'] = df['주차']/24 #6개월(4주*6)
        # print(df['나이'].value_counts())

        # df['날짜'] = pd.to_datetime(df['일']).dt.strftime('%m%d')
        # df['날짜'] = df['날짜'].astype(str)
        # df['날짜'] = df['날짜'].astype(int)
        # print(df['날짜'].value_counts())

        df['개화적온'] = 0
        df.loc[(df['내부온도']>=20) & (df['내부온도']<=25), '개화적온'] = 1
        # print(df['개화적온'].value_counts())

        df['적정습도'] = 0
        df.loc[(df['내부습도']>=65) & (df['내부습도']<=80), '적정습도'] = 1
        # print(df['적정습도'].value_counts())

        df['적정CO2'] = 0
        df.loc[(df['내부CO2']>=400) & (df['내부습도']<=700), '적정CO2'] = 1
        # print(df['적정CO2'].value_counts())

        df['적정온도'] = 0
        df.loc[(df['내부온도']>=20) & (df['내부온도']<=25), '적정온도'] = 1
        # print(df['적정온도'].value_counts())

        df['고습'] = 0
        df.loc[(df['내부습도']>85), '고습'] = 1
        # print(df['고습'].value_counts())

        df['고온'] = 0
        df.loc[(df['고온']>28), '고온'] = 1
        # print(df['고온'].value_counts())

        df['건조'] = 0
        df.loc[(df['내부습도']<40), '건조'] = 1
        # print(df['건조'].value_counts())

        df['냉해'] = 0
        df.loc[(df['내부온도']<10), '냉해'] = 1
        # print(df['냉해'].value_counts())
        #
        # df['급액EC단계'] = 0
        # df.loc[(df['급액EC(dS/m)']>0) & (df['급액EC(dS/m)']<=0.8), '급액EC단계'] = 1
        # df.loc[(df['급액EC(dS/m)']>0.8) & (df['급액EC(dS/m)']<=1.5), '급액EC단계'] = 2
        # df.loc[(df['급액EC(dS/m)']>1.5), '급액EC단계'] = 3
        # print(df['급액EC단계'].value_counts())

        df['분기'] = 0
        df.loc[(df['일(월)']>=0) & (df['급액EC(dS/m)']<4), '분기'] = 3
        df.loc[(df['일(월)']>=4) & (df['급액EC(dS/m)']<7), '분기'] = 2
        df.loc[(df['일(월)']>=7) & (df['급액EC(dS/m)']<10), '분기'] = 1
        df.loc[(df['일(월)']>=10) & (df['급액EC(dS/m)']<13), '분기'] = 4
        # print(df['분기'].value_counts())


        return df

    def integrate_input_df_by_samples(self, df_in, df_out=None, method='sum'):
        df_in['일'] = 0

        print(df_in.columns.tolist())
        if method == 'sum':
            df_in = df_in.groupby(['Sample_no'], as_index=False).sum()

            # 합하면 안되는 값은 원위치로
            for col in ['시설ID', '일', '주차', '내부CO2', '내부습도', '내부온도', '일사량', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)', '총급여량', '나이']:
            # for col in ['주차', '품종', '재배형태', '나이', '일(연)', '일(월)', '일(주)', '일(일)', '분기', '시설ID']:
                df_in[col] = df_in[col]/7

        if method == 'mean':
            df_in = df_in.groupby(['Sample_no'], as_index=False).mean()


        df_in_dup = df_in.iloc[:, 1:].duplicated()
        # print(df_in)
        # print(df_in_dup)
        # 중복 아닌 데이터프레임
        idx_dup = df_in_dup[df_in_dup==True].index.tolist()
        sample_dup = [df_in.iloc[idx, 0] for idx in idx_dup]

        # df_in에서 중복되는 샘플제거
        df_in = df_in.drop(idx_dup, axis=0)
        # print(idx_dup)
        # print(sample_dup)
        # 중복되는 샘플정보들을 output에서 삭제

        if df_out is not None:
            print(df_out.shape)
            row_drop = []
            for idx in df_out.index.tolist():

                sample_no = df_out.iloc[idx, 0]
                if sample_no in sample_dup:
                    row_drop.append(idx)

            df_out.drop(row_drop, axis=0, inplace=True)

        return df_in, df_out

    def integrate_test_input_df_by_samples(self, df_in, df_out=None, method='sum'):
        df_in['일'] = 0

        print(df_in.columns.tolist())
        if method == 'sum':
            df_in = df_in.groupby(['Sample_no'], as_index=False).sum()

            # 합하면 안되는 값은 원위치로
            for col in ['시설ID', '일', '주차', '내부CO2', '내부습도', '내부온도', '일사량', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)', '품종', '재배형태', '일(연)', '일(월)', '일(주)', '일(일)', '총급여량', '나이']:
            # for col in ['주차', '품종', '재배형태', '나이', '일(연)', '일(월)', '일(주)', '일(일)', '분기', '시설ID']:
                df_in[col] = df_in[col]/7

        if method == 'mean':
            df_in = df_in.groupby(['Sample_no'], as_index=False).mean()


        # df_in_dup = df_in.iloc[:, 1:].duplicated()
        # # print(df_in)
        # # print(df_in_dup)
        # # 중복 아닌 데이터프레임
        # idx_dup = df_in_dup[df_in_dup==True].index.tolist()
        # sample_dup = [df_in.iloc[idx, 0] for idx in idx_dup]
        #
        # # df_in에서 중복되는 샘플제거
        # df_in = df_in.drop(idx_dup, axis=0)
        # # print(idx_dup)
        # # print(sample_dup)
        # # 중복되는 샘플정보들을 output에서 삭제
        #
        # if df_out is not None:
        #     print(df_out.shape)
        #     row_drop = []
        #     for idx in df_out.index.tolist():
        #
        #         sample_no = df_out.iloc[idx, 0]
        #         if sample_no in sample_dup:
        #             row_drop.append(idx)
        #
        #     df_out.drop(row_drop, axis=0, inplace=True)

        return df_in, df_out


    def one_hot_encoding(self, df, list_columns):
        df = pd.get_dummies(df, columns = list_columns)

        return df


