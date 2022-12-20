import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dateutil import relativedelta
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV
import streamlit as st
import altair as alt


#Yahoo Finaceでデータ取得するためのtickerを辞書登録
tickers = {
    'トヨタ': '7203.T',
    '任天堂': '7974.T',
    'ソニー': '6758.T',
}


# サイドバーの設定
# 企業名の選択
company = st.sidebar.selectbox('銘柄を選択してください',['トヨタ', 'ソニー','任天堂'])
# 会社名選択
selectmodel = st.sidebar.selectbox('予測モデルを選択してください',['決定木モデル','重回帰モデル'])

# 訓練データの期間(1年～３０年で設定。標準を１０年分とする)
span_years = st.sidebar.slider("学習データの期間", 1 ,30 ,10)
today = datetime.now()
yesterday = today - relativedelta.relativedelta(days=1)
EndDay = today - relativedelta.relativedelta(months=1)
StartDay = EndDay - relativedelta.relativedelta(years=span_years)


# 日経平均株価の取得
nikkei = yf.Ticker("^N225")
Start = today - relativedelta.relativedelta(years=31)
hist_nikkei = nikkei.history(start=Start , end=today)
hist_nikkei = hist_nikkei.reset_index()

# 標準時刻部分の削除
hist_nikkei["Date"] = hist_nikkei["Date"].dt.tz_localize(None)
hist_nikkei = hist_nikkei.drop(columns=["Dividends","Stock Splits"])

# 日経平均の学習データを作成
train_nikkei = hist_nikkei[(hist_nikkei["Date"] > StartDay) & (hist_nikkei["Date"] < EndDay)]
# train_nikkei
# 日経平均の評価データを作成
test_nikkei = hist_nikkei[(hist_nikkei["Date"] > EndDay) & (hist_nikkei["Date"] < today)]
# test_nikkei



# 選択した企業の学習データを作成
STOCK = yf.Ticker(tickers[company])

# とりあえず、直近３１年のデータだけに絞る。
today = datetime.now()
Start =today - relativedelta.relativedelta(years=31)
hist_STOCK = STOCK.history(start=Start , end=today)

hist_STOCK = hist_STOCK.reset_index()

# 標準時刻部分の削除
hist_STOCK["Date"] = hist_STOCK["Date"].dt.tz_localize(None)
hist_STOCK["Weekday"] = [x.weekday() for x in hist_STOCK["Date"]]
hist_STOCK = hist_STOCK.drop(columns=["Dividends","Stock Splits"])
hist_STOCK["Up"] = hist_STOCK["Close"].diff(-1).apply(lambda x :0 if x > 0 else 1)

# 学習データの作成
train = hist_STOCK[(hist_STOCK["Date"] > StartDay) & (hist_STOCK["Date"] < EndDay)]
# 評価データの作成
test = hist_STOCK[(hist_STOCK["Date"] > EndDay) & (hist_STOCK["Date"] < today)]

# 企業データと日経平均を組み合わせ、学習/評価データをそれぞれ作成する
trainX = pd.merge(train,train_nikkei,how="left",on="Date").drop(columns=["Date","Up"])
trainX = trainX.interpolate(limit_direction='both')

testX = pd.merge(test,test_nikkei,how="left",on="Date").drop(columns=["Date","Up"])
testX = testX.interpolate(limit_direction='both')

y = train["Up"][(hist_STOCK["Date"] > StartDay) & (hist_STOCK["Date"] < EndDay)]

# ダミー関数化
trainX = pd.get_dummies(trainX)
testX = pd.get_dummies(testX)


if selectmodel == "重回帰モデル":
    st.sidebar.text("作成中")
    text = "作成"
    CorrectRate = 0

else:
    clf = DT()
    parameters = {"max_depth":list(range(2,11)),"min_samples_leaf":[5,10,20,50,60,70,80,90,100,500]}
    gcv = GridSearchCV(clf, parameters, cv=5, scoring="roc_auc",n_jobs=-1,return_train_score=True)
    gcv.fit(trainX,y)

    pred = gcv.predict_proba(testX)
    pred = pred[:,1]

    # 結果評価作成
    sample = test[["Date","Close","Up"]].copy()

    sample["予測"] = pred
    # sample["予測確率"] = sample["予測"].apply(lambda x : 1 if x > 0.5 else 0)
    sample["予測"] = sample["予測"].apply(lambda x : 1 if x > 0.5 else 0)
    sample["正誤"] = sample["予測"] - sample["Up"]
    sample["正誤"] = sample["正誤"].apply(lambda x : "〇" if x == 0 else "×")

    sample["Up"] = sample["Up"].apply(lambda x : "UP" if x == 1 else "DOWN")
    sample["予測"] = sample["予測"].apply(lambda x : "UP" if x > 0.5 else "DOWN")


    # 正答率の算出
    df_bool = (sample["正誤"] == "〇")
    CorrectRate = df_bool.sum() / len(sample["正誤"]) * 100

    text = sample["予測"].iloc[-0]
    # text



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
    df = get_data(days, tickers)

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
