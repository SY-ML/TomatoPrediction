from tomato_preprocess import PreProcess
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


class Visualize(PreProcess):
    def __init__(self, data_directory, outputBy_colName):
        super().__init__(data_directory, outputBy_colName)

    def show_scatter_controlled_columns_by(self, df, col, title=f"INPUT SCATTER"):
        title = title+f"by {col}"
        # train = pp.merge_in
        df = df.loc[df[col].notnull()]
        df = df.dropna(axis=1)
        cols = ['내부CO2', '내부습도', '내부온도', '일사량', '급액횟수', '급액EC(dS/m)', '급액pH', '급액량(회당)']
        f, ax = plt.subplots(2, 4, figsize=(18, 10))
        ax[0][0].scatter(data=df, x=col, y='내부CO2')
        ax[0][0].set_title("내부CO2")

        ax[0][1].scatter(data=df, x=col, y='내부습도')
        ax[0][1].set_title("내부습도")
        ax[0][2].scatter(data=df, x=col, y='내부온도')
        ax[0][2].set_title("내부온도")
        ax[0][3].scatter(data=df, x=col, y='일사량')
        ax[0][3].set_title("일사량")
        ax[1][0].scatter(data=df, x=col, y='급액횟수')
        ax[1][0].set_title("급액횟수")
        ax[1][1].scatter(data=df, x=col, y='급액EC(dS/m)')
        ax[1][1].set_title("급액EC(dS/m)")
        ax[1][2].scatter(data=df, x=col, y='급액pH')
        ax[1][2].set_title("급액pH")
        ax[1][3].scatter(data=df, x=col, y='급액량(회당)')
        ax[1][3].set_title("급액량(회당)")
        ax[0][0].grid()
        ax[0][1].grid()
        ax[0][2].grid()
        ax[0][3].grid()
        ax[1][0].grid()
        ax[1][1].grid()
        ax[1][2].grid()
        ax[1][3].grid()

        f.suptitle(title)
        plt.show()

    def show_scatter_outputs(self, df_out, title = "TRAIN_OUTPUT BY 주차"):
        print(f"========={title}=========")
        print(df_out.describe())
        f, ax = plt.subplots(1, 3, figsize=(18, 10))
        col = '주차'
        f.suptitle(title)
        ax[0].scatter(data=df_out, x=col, y='생장길이')
        ax[0].set_title(f"생장길이 by{col}")
        ax[1].scatter(data=df_out, x=col, y='줄기직경')
        ax[1].set_title(f"줄기직경 by{col}")
        ax[2].scatter(data=df_out, x=col, y='개화군')
        ax[2].set_title(f"개화군 by{col}")
        plt.show()
    #
    # def show_scatter_outputs_by(self, df_in, df_out, col):
    #     group_sample = df_in.groupby(['Sample_no'], as_index=False).mean()
    #     train_out = pd.merge(df_out, group_sample, on='Sample_no')
    #
    #     f, ax = plt.subplots(1, 3, figsize=(18, 10))
    #     # x = '주차'
    #     ax[0].scatter(data=train_out, x=col, y='생장길이')
    #     ax[0].set_title(f"생장길이 by{col}")
    #     ax[1].scatter(data=train_out, x=col, y='줄기직경')
    #     ax[1].set_title(f"줄기직경 by{col}")
    #     ax[2].scatter(data=train_out, x=col, y='개화군')
    #     ax[2].set_title(f"개화군 by{col}")
    #     plt.show()
