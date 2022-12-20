import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
import calendar
from dateutil import relativedelta
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV
import streamlit as st
import altair as alt


#Yahoo Finaceでデータ取得するためのtickerを辞書登録
BrandTickers = {
    'トヨタ': '7203.T',
    '任天堂': '7974.T',
    'ソニー': '6758.T',
}

# TrainningTickers = {
#     '日経平均': '^N225'
# }
TrainningTickers = {}


def SpanTime(span_years):
    # 取得するデータの期間を設定    
    today = datetime.now()
    # Start = today - relativedelta.relativedelta(years=12)               # 取得する全データの期間 31年前からで仮設定

    testEndday = today - relativedelta.relativedelta(days=1)            # ”昨日”
    testStartDay = testEndday - relativedelta.relativedelta(months=1)   # 評価データ期間（直近１か月）

    if 1 <= today.month < 4 :
        trainMonth = 9
        trainYear = today.year - 1
    elif 4 <= today.month < 10 :
        trainMonth = 3
        trainYear = today.year
    else:
        trainMonth = 9
        trainYear = today.year

    trainEndDay = datetime(trainYear,trainMonth, calendar.monthrange(trainYear, trainMonth)[1])
    trainStartDay = trainEndDay - relativedelta.relativedelta(years=span_years+1)   # 学習データ期間（設定期間）

    return trainStartDay,trainEndDay,testStartDay,testEndday,today


# 株価取得関数
@st.cache
def get_STOCK_Data(company):
    STOCK = yf.Ticker(BrandTickers[company])

    # 期間の設定
    trainStartDay,trainEndDay,testStartDay,testEndday,today = SpanTime(span_years)

    # 全データの取得
    STOCK_hist = STOCK.history(start=trainStartDay , end=today)
    STOCK_hist = STOCK_hist.reset_index()

    # 標準時刻部分の削除
    STOCK_hist["Date"] = STOCK_hist["Date"].dt.tz_localize(None)
    # STOCK_hist = STOCK_hist.drop(columns=["Dividends","Stock Splits"])
    STOCK_hist = STOCK_hist.drop(columns=["Volume","Dividends","Stock Splits"])

    # 曜日と上/下の追加
    STOCK_hist["Weekday"] = [x.weekday() for x in STOCK_hist["Date"]]
    STOCK_hist["Up"] = STOCK_hist["Close"].diff(-1).apply(lambda x :0 if x > 0 else 1)

    #過去２週間分のデータを後ろに設置
    STOCK_hist["delta_High"] = STOCK_hist["High"] - STOCK_hist["Open"]
    STOCK_hist["delta_Low"] = STOCK_hist["Low"] - STOCK_hist["Open"]
    STOCK_hist["delta_Close"] = STOCK_hist["Close"] - STOCK_hist["Open"]
    STOCK_hist["delta_High-Low"] = STOCK_hist["High"] - STOCK_hist["Low"]

    STOCK_hist["delta_High_rate"] = (STOCK_hist["High"] - STOCK_hist["Open"])/STOCK_hist["Open"]*100
    STOCK_hist["delta_Low_rate"] = (STOCK_hist["Low"] - STOCK_hist["Open"])/STOCK_hist["Open"]*100
    STOCK_hist["delta_Close_rate"] = (STOCK_hist["Close"] - STOCK_hist["Open"])/STOCK_hist["Open"]*100
    STOCK_hist["delta_High-Low_rate"] = (STOCK_hist["High"] - STOCK_hist["Low"])/STOCK_hist["Open"]*100

    STOCK_hist["delta"] = STOCK_hist["Close"].diff(1)

    STOCK_hist_base = STOCK_hist.copy()

    for i in range(1,15):
        STOCK_histXp = STOCK_hist_base.copy()
        STOCK_histXp = STOCK_histXp.drop(columns="Weekday")
        STOCK_histXp["Date"] = STOCK_histXp["Date"] + timedelta(days=i)
        STOCK_histXp_header = STOCK_histXp.columns
        STOCK_histXp_header = [ x + "_P" + str(i) for x in STOCK_histXp_header ]
        STOCK_histXp_header[0] = "Date"
        STOCK_histXp.columns = STOCK_histXp_header

        STOCK_hist = pd.merge(STOCK_hist,STOCK_histXp,how="left")

    # 学習データを作成
    train_STOCK = STOCK_hist[(STOCK_hist["Date"] > trainStartDay) & (STOCK_hist["Date"] < trainEndDay)]
    # 評価データを作成
    test_STOCK = STOCK_hist[(STOCK_hist["Date"] > testStartDay) & (STOCK_hist["Date"] < testEndday)]
    # 本日のデータを作成
    today_STOCK = STOCK_hist[STOCK_hist["Date"] == today]

    return train_STOCK, test_STOCK, today_STOCK


# 説明変数取得関数
@st.cache
def get_TRAINNING_Data(TrainningTickers):

    # 期間の設定
    trainStartDay,trainEndDay,testStartDay,testEndday,today = SpanTime(span_years)
    # 日経平均をもとにする
    STOCK = yf.Ticker("^N225")

    # 全データの取得
    STOCK_hist = STOCK.history(start=trainStartDay , end=today)
    STOCK_hist = STOCK_hist.reset_index()

    # 標準時刻部分の削除
    STOCK_hist["Date"] = STOCK_hist["Date"].dt.tz_localize(None)
    # STOCK_hist = STOCK_hist.drop(columns=["Dividends","Stock Splits"])
    STOCK_hist = STOCK_hist.drop(columns=["Volume","Dividends","Stock Splits"])

    # カラム名の変更(mergeするときに同じになってしまうので)
    STOCK_hist_header = STOCK_hist.columns
    STOCK_hist_header = [ x + "_^N225" for x in STOCK_hist_header ]
    STOCK_hist_header[0] = "Date"
    STOCK_hist.columns = STOCK_hist_header

    # 選択した説明変数の組み込み
    for value in TrainningTickers.values():
        STOCK = yf.Ticker(value)

        # 全データの取得
        STOCK_histP = STOCK.history(start=trainStartDay , end=today)
        STOCK_histP = STOCK_histP.reset_index()

        # 標準時刻部分の削除
        STOCK_histP["Date"] = STOCK_histP["Date"].dt.tz_localize(None)
        # STOCK_histP = STOCK_histP.drop(columns=["Dividends","Stock Splits"])
        STOCK_histP = STOCK_histP.drop(columns=["Volume","Dividends","Stock Splits"])

        # カラム名の変更(mergeするときに同じになってしまうので)
        STOCK_hist_header = STOCK_histP.columns
        STOCK_hist_header = [ x + "_" + value for x in STOCK_hist_header ]
        STOCK_hist_header[0] = "Date"
        STOCK_histP.columns = STOCK_hist_header

        STOCK_hist = pd.merge(STOCK_hist,STOCK_histP,how="left",on="Date")

    # 学習データを作成
    train_STOCK = STOCK_hist[(STOCK_hist["Date"] > trainStartDay) & (STOCK_hist["Date"] < trainEndDay)]
    # 評価データを作成
    test_STOCK = STOCK_hist[(STOCK_hist["Date"] > testStartDay) & (STOCK_hist["Date"] < testEndday)]
    # 今日のデータを作成
    today_STOCK = STOCK_hist[STOCK_hist["Date"] == today]

    return train_STOCK, test_STOCK, today_STOCK

@st.cache
# 決定木モデルの作成
def makeTreeModel(clf,trainX,y):
    parameters = {"max_depth":list(range(2,11)),"min_samples_leaf":[5,10,20,50,60,70,80,90,100,500]}
    gcv = GridSearchCV(clf, parameters, cv=5, scoring="roc_auc",n_jobs=-1)
    gcv.fit(trainX,y)

    return gcv

# サイドバーの設定
# 企業名の選択
company = st.sidebar.selectbox('銘柄を選択してください',['トヨタ', 'ソニー','任天堂'])
# 会社名選択
selectmodel = st.sidebar.selectbox('予測モデルを選択してください',['決定木モデル','重回帰モデル'])

# # 訓練データの期間(1年～３０年で設定。標準を１０年分とする)
span_years = st.sidebar.slider("学習データの期間", 1 ,10 ,10)

text = "算出"
CorrectRate = 0
sample = []

if selectmodel == "決定木モデル":

    # 説明変数の選択
    st.sidebar.text("説明変数の選択して下さい")
    st.sidebar.text("アメリカ系")
    if st.sidebar.checkbox("NYダウ",value=True):
        TrainningTickers["NYダウ"] = "^DJI"

    # if st.sidebar.checkbox("ナスダック総合"):
    #     TrainningTickers["ナスダック総合"] = "^IXIC"
    # if st.sidebar.checkbox("S＆P500"):
    #     TrainningTickers["S＆P500"] = "^GSPC"

    if st.sidebar.checkbox("米10年国債"):
        TrainningTickers["米10年国債"] = "^TNX"

    if st.sidebar.checkbox("ドル（アメリカ）"):
        TrainningTickers["ドル（アメリカ）"] = "USDJPY=X"

    st.sidebar.text("セクター別S&P500指数")
    if st.sidebar.checkbox("エネルギー"):
        TrainningTickers["エネルギー"] = "^GSPE"
    
    if st.sidebar.checkbox("工業・資本財サービス"):
        TrainningTickers["工業・資本財サービス"] = "^SP500-20"
    
    if st.sidebar.checkbox("生活必需品"):
        TrainningTickers["生活必需品"] = "^SP500-30"

    if st.sidebar.checkbox("金融"):
        TrainningTickers["金融"] = "^SP500-40"

    if st.sidebar.checkbox("電気通信サービス"):
        TrainningTickers["電気通信サービス"] = "^SP500-50"
    
    if st.sidebar.checkbox("素材"):
        TrainningTickers["素材"] = "^SP500-15"

    if st.sidebar.checkbox("一般消費財・サービス"):
        TrainningTickers["一般消費財・サービス"] = "^SP500-25"

    if st.sidebar.checkbox("ヘルスケア"):
        TrainningTickers["ヘルスケア"] = "^SP500-35"

    if st.sidebar.checkbox("情報技術"):
        TrainningTickers["情報技術"] = "^SP500-45"

    if st.sidebar.checkbox("公共事業"):
        TrainningTickers["公共事業"] = "^SP500-55"
    

    if st.sidebar.button("モデルを作成"):

        # 選択した企業の学習データを作成
        train_BrandStock, test_BrandStock, today_BrandStock = get_STOCK_Data(company)
        # 正解データの作成
        y = train_BrandStock["Up"]

        # 説明変数の取得
        trainStock, testStock, todayStock = get_TRAINNING_Data(TrainningTickers)

        # 企業データと日経平均を組み合わせ、学習/評価データをそれぞれ作成する
        # Date列があると線形補正できない為、Date列を削除
        trainX = pd.merge(train_BrandStock,trainStock,how="left",on="Date").drop(columns=["Date","Up"])
        trainX = trainX.interpolate(limit_direction='both')

        testX = pd.merge(test_BrandStock,testStock,how="left",on="Date").drop(columns=["Date","Up"])
        testX = testX.interpolate(limit_direction='both')

        todayX = pd.merge(today_BrandStock,todayStock,how="left",on="Date").drop(columns=["Date","Up"])

        # ダミー関数化
        trainX = pd.get_dummies(trainX)
        testX = pd.get_dummies(testX)
        todayX = pd.get_dummies(todayX)

        clf = DT()
        # 決定木モデルの作成
        gcv = makeTreeModel(clf,trainX,y)
        pred = gcv.predict_proba(testX)
        pred = pred[:,1]

        # 結果評価作成
        sample = test_BrandStock[["Date","Close","Up"]].copy()
        sample["予測"] = pred
        sample["予測"] = sample["予測"].apply(lambda x : 1 if x > 0.5 else 0)
        sample["正誤"] = sample["予測"] - sample["Up"]
        sample["正誤"] = sample["正誤"].apply(lambda x : "〇" if x == 0 else "×")

        sample["Up"] = sample["Up"].apply(lambda x : "UP" if x == 1 else "DOWN")
        sample["予測"] = sample["予測"].apply(lambda x : "UP" if x > 0.5 else "DOWN")
        sample["確率"] = list(map( lambda x : x*100  if x > 0.5 else (1-x)*100, pred))

        # 正答率の算出
        df_bool = (sample["正誤"] == "〇")
        CorrectRate = df_bool.sum() / len(sample["正誤"]) * 100

        text = sample["予測"].iloc[-0]
        # text



# elif selectmodel == "重回帰モデル":
#     #モデル生成

#     def CreateWeek(date):
#         return date.strftime('%a')

#     date = pd.to_datetime(train["Date"])
#     train["week"] = date.apply(lambda x:CreateWeek(x))

#     date = pd.to_datetime(test["Date"])
#     test["week"] = date.apply(lambda x:CreateWeek(x))

#     trainX = pd.get_dummies(train[["Stock_High","Stock_Close","Exchange_High","Exchange_Close","week"]])

#     y = train["Up"]

#     model = LR()

#     model.fit(trainX,y)

#     testX = pd.get_dummies(test[["Stock_High","Stock_Close","Exchange_High","Exchange_Close","week"]])

#     pred = model.predict(testX)

#     pred = np.where(pred >=1 , "UP", "DOWN")
#     test["pred"] = pred

#     result = np.diff(test["Stock_Close"])
#     result = np.where(result >1, "UP","DOWN")
#     result = np.append(result,"結果待ち")
#     test["result"] = result

#     assessment = np.where(pred == result, "〇","×")
#     test["assessment"] = assessment

#     #日付フォーマットを修正
#     test["Date"] = pd.to_datetime(test["Date"])
#     test["Date"] =test["Date"].dt.strftime('%Y/%m/%d')
#     sample["Date"] = pd.to_datetime(sample["Date"])
#     sample["Date"] =sample["Date"].dt.strftime('%Y/%m/%d')

#     #sampleファイルを作成
#     merge = pd.merge(sample,test, on="Date")
#     sample = merge.drop(["Stock_Open","Stock_High","Stock_Low","Exchange_Open","Exchange_High","Exchange_Low","Exchange_Close","week"], axis=1)
#     sample = sample.rename(columns={'Date':'日付','Stock_Close':'終値','pred':'予測','result':'結果','assessment':'正誤'})

#     #終値を3桁カンマ表示に修正
#     close = sample["終値"] 
#     close = close.astype("int64")
#     sample["終値"]  = close.map('{:,}'.format)

#     text = sample["予測"].iloc[-0]

#     pass


else:
    pass


# メイン画面作成
# タイトル
st.title('6班キンカク 株価予測アプリ')
st.write("翌取引日の終値は...") 
st.info(f"{text}します！")
tab1, tab2 = st.tabs(["🗃 モデルの成績","📈株価推移"])


with tab1:

    st.write("#### モデルの正解率（直近30日間）")
    st.write(f"{CorrectRate:.1f}%")

    st.write("#### 直近30日間（詳細）")
    df = st.dataframe(sample)


with tab2:

    days = 100

    #関数「表データの生成」
    @st.cache                                                   # 毎回とってくるのではなくcacheに溜めることで処理を高速化する
    def get_data(days, tickers):
        df = pd.DataFrame()
        for company in tickers.keys():
            tkr = yf.Ticker(tickers[company])                   # Yahoo Financeから情報を持ってくる
            hist = tkr.history(period=f'{days}d')               # 抽出期間の指定
            hist.index = hist.index.strftime('%d %B %Y')        # 日付フォーマット変更
            hist = hist[['Close']]                              # 終値のみを使う
            hist.columns = [company]                            # カラム名を社名とする
            hist = hist.T                                       # 行列を反転する
            hist.index.name = 'Name'                            # 1列目の列名を"Name"とする
            df = pd.concat([df, hist])                          # 横に連結（appleの横にAmazon等）
        return df

    if company == "トヨタ":            
        ymin, ymax = 1000,3000
    elif company == "任天堂":
        ymin, ymax = 4500,6500
    elif company == "ソニー":
        ymin, ymax = 8500,13000

    #関数呼び出し
    df = get_data(days, BrandTickers)

    #メイン(4) 株価の表とグラフを表示
    data = df.loc[company]                            # 指定企業のみを抽出
    data = data.T.reset_index()                         # [グラフのために整形] 転値
    data = pd.melt(data, id_vars=['Date']).rename(      # [グラフのために整形] 説明難しいが、ピボットテーブルの逆（集計→生にmeltした）
        columns={'value': '株価（円）'}                 # [グラフのために整形] カラム名を変更
    )
    chart = (
        alt.Chart(data)                                                                         # dataを使ってグラフを作成
        .mark_line(opacity=0.8, clip=True)                                                      # 折れ線グラフ（opacityは透明度、Clip=Trueは上限下限値意外は表示しないという意味）
        .encode(
            x="Date:T",                                                                         # X軸は日付
            y=alt.Y("株価（円）:Q", stack=None, scale=alt.Scale(domain=[ymin, ymax])),   # Y軸のカラム名を設定
            color='Name:N'
        )
    )
    st.altair_chart(chart, use_container_width=True)    # グラフを表示（use_container_width=Trueは枠にチャートを表示する）        except:
