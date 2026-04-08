import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import mean_absolute_error

#Step 1 Pengumpulan Data
issiPath = "file:///C:/papervolatilitas/issi.csv"
jiiPath  = "file:///C:/papervolatilitas/JII.csv"

issiData = pd.read_csv(issiPath)
jiiData  = pd.read_csv(jiiPath)

#Step 2 Praproses Data

#konversi tanggal menjadi date dan terakhir menjadi close 
issiData.rename(columns={'Tanggal': 'Date', 'Terakhir': 'Close'}, inplace=True)
jiiData.rename(columns={'Tanggal': 'Date', 'Terakhir': 'Close'}, inplace=True)

#membuat function untuk preprocessing data
def preprocessData(df):
    #merubah format tanggal menjadi datetime milik dataframe dan dayfirst = true berarti bagian awal itu adalah angka hari bukan bulan
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    #konversi angka harga closing menjadi float 
    df['Close'] = (
        df['Close']
        .astype(str)
        .str.replace(r'\.', '', regex=True)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    return df

issiData = preprocessData(issiData)
jiiData  = preprocessData(jiiData)

# Samakan tanggal
commonDates = issiData.index.intersection(jiiData.index)
issiData = issiData.loc[commonDates]
jiiData  = jiiData.loc[commonDates]

# Menampilkan grafik harga penutupan dari data harian
plt.figure(figsize=(11,5))
plt.plot(issiData.index, issiData['Close'], label='ISSI', linewidth=2)
plt.plot(jiiData.index, jiiData['Close'], label='JII', linewidth=2, linestyle='--')
plt.title("Perbandingan Harga Asli Indeks ISSI dan JII (Data Harian)")
plt.xlabel("Tanggal")
plt.ylabel("Harga Indeks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Step 3 Log Return Mingguan 

#Resample w itu berfungsi mengelompokkan data closingan yang sudah jadi date harganya dikelompokkan menjadi mingguan dengan mengambil data terakhir di minggu tersebut karna syntax last()
issiWeekly = issiData['Close'].resample('W').last()
jiiWeekly  = jiiData['Close'].resample('W').last()


#Rumus Log Return Mingguan =LN(Current_Price / Previous_Price). 
issiLogReturn = np.log(issiWeekly / issiWeekly.shift(1)).dropna()
jiiLogReturn  = np.log(jiiWeekly / jiiWeekly.shift(1)).dropna()
weeklyLogReturns = pd.DataFrame({
    'ISSI': issiLogReturn,
    'JII': jiiLogReturn
})

#tampilin dataframenya
print(weeklyLogReturns.head())

plt.figure(figsize=(11,5))
plt.plot(weeklyLogReturns.index, weeklyLogReturns['ISSI'],label='ISSI', linewidth=1.5)
plt.plot(weeklyLogReturns.index, weeklyLogReturns['JII'],label='JII', linewidth=1.5, linestyle='--')
plt.title("Log Return Mingguan Indeks ISSI dan JII")
plt.xlabel("Tanggal")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#rolling window 4 minggu
window = 4 

#rumus volatilitas = standar deviasi dari hasil log return dikali akar 4 karena windownya 4 week sesuai aturan square root of time rule in finance
issiVolatility = weeklyLogReturns['ISSI'].rolling(window=window).std() * np.sqrt(window)
jiiVolatility  = weeklyLogReturns['JII'].rolling(window=window).std() * np.sqrt(window)
weeklyVolatility = pd.DataFrame({
    'ISSI Volatility': issiVolatility,
    'JII Volatility': jiiVolatility
})

#tampilin dataframenya
print(weeklyVolatility.head())

plt.figure(figsize=(11,5))
plt.plot(weeklyVolatility.index, weeklyVolatility['ISSI Volatility'],label='ISSI', linewidth=1.5)
plt.plot(weeklyVolatility.index, weeklyVolatility['JII Volatility'],label='JII', linewidth=1.5, linestyle='--')
plt.title("Log Return Mingguan Indeks ISSI dan JII")
plt.xlabel("Tanggal")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def grafikDatTesting(volatilityIndex, title):
    dfPlot = volatilityIndex.dropna().to_frame(name='Actual')

    #lag-1 berarti mengambil nilai 1 periode sebelumnya


    #prediksi hari ini = minggu ke 2 = minggu ke 1 
    #minggu ke 3 = minggu ke 2
    #Prediksi Statistik (Naïve Lag-1)
    dfPlot['predStat'] = dfPlot['Actual'].shift(1)

    #ML Fitur Lag-1
    dfPlot['lag1'] = dfPlot['Actual'].shift(1)
    dfPlot.dropna(inplace=True)
    X = dfPlot[['lag1']]
    y = dfPlot['Actual']

    #Split data 70% data belajar 20% data uji
    splitIdx = int(len(dfPlot) * 0.7)
    XTrain, XTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    #Training Model ML
    rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
    rfModel.fit(XTrain, yTrain)
    yPredMl = rfModel.predict(XTest)

    #Ambil data testing
    dfTest = dfPlot.loc[XTest.index].copy()
    dfTest['predMl'] = yPredMl

    #Grafik
    plt.figure(figsize=(12,5))
    plt.plot(dfTest.index, dfTest['Actual'], label='Actual Volatility', linewidth=2)
    plt.plot(dfTest.index, dfTest['predMl'], linestyle='--', label='Predicted ML')
    plt.plot(dfTest.index, dfTest['predStat'], linestyle=':', label='Predicted Statistik (Lag-1)')
    plt.title(title)
    plt.xlabel("Tanggal")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


grafikDatTesting(
    weeklyVolatility['JII Volatility'],
    "Perbandingan Prediksi Volatilitas JII"
)

grafikDatTesting(
    weeklyVolatility['ISSI Volatility'],
    "Perbandingan Prediksi Volatilitas ISSI"
)



# Evaluasi RMSE dan MAE
pandemiStart = '2020-01-01'
pandemiEnd   = '2023-06-30'

postPandemiStart = '2023-07-01'
postPandemiEnd   = weeklyVolatility.index.max()  # data terakhir

pandemiMask = (weeklyVolatility.index >= pandemiStart) & (weeklyVolatility.index <= pandemiEnd)
postMask    = (weeklyVolatility.index >= postPandemiStart) & (weeklyVolatility.index <= postPandemiEnd)


def computeRmseMl(volSeries, periodMask):
    dfMl = volSeries.loc[periodMask].copy().to_frame(name='Vol')
    dfMl['lag1'] = dfMl['Vol'].shift(1)
    dfMl.dropna(inplace=True)

    X = dfMl[['lag1']]
    y = dfMl['Vol']

    splitIdx = int(len(dfMl) * 0.9)
    XTrain, XTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
    rfModel.fit(XTrain, yTrain)
    yPred = rfModel.predict(XTest)

    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    return rmse


def computeRmseStatistic(volSeries, periodMask):
    dfStat = volSeries.loc[periodMask].copy()
    dfStatLag = dfStat.shift(1)

    validIdx = dfStat.index.intersection(dfStatLag.dropna().index)
    dfStat = dfStat.loc[validIdx]
    dfStatLag = dfStatLag.loc[validIdx]

    rmseStat = np.sqrt(mean_squared_error(dfStat, dfStatLag))
    return rmseStat


def computeMaeMl(volSeries, periodMask):
    dfMl = volSeries.loc[periodMask].copy().to_frame(name='Vol')
    dfMl['lag1'] = dfMl['Vol'].shift(1)
    dfMl.dropna(inplace=True)

    X = dfMl[['lag1']]
    y = dfMl['Vol']

    splitIdx = int(len(dfMl) * 0.9)
    XTrain, XTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
    rfModel.fit(XTrain, yTrain)
    yPred = rfModel.predict(XTest)

    return mean_absolute_error(yTest, yPred)


def computeMaeStatistic(volSeries, periodMask):
    dfStat = volSeries.loc[periodMask].copy()
    dfStatLag = dfStat.shift(1)

    validIdx = dfStat.index.intersection(dfStatLag.dropna().index)
    dfStat = dfStat.loc[validIdx]
    dfStatLag = dfStatLag.loc[validIdx]

    return mean_absolute_error(dfStat, dfStatLag)


rmseJiiStatPandemi  = computeRmseStatistic(weeklyVolatility['JII Volatility'], pandemiMask)
rmseJiiStatPost     = computeRmseStatistic(weeklyVolatility['JII Volatility'], postMask)
rmseIssiStatPandemi = computeRmseStatistic(weeklyVolatility['ISSI Volatility'], pandemiMask)
rmseIssiStatPost    = computeRmseStatistic(weeklyVolatility['ISSI Volatility'], postMask)

maeJiiMlPandemi  = computeMaeMl(weeklyVolatility['JII Volatility'], pandemiMask)
maeJiiMlPost     = computeMaeMl(weeklyVolatility['JII Volatility'], postMask)
maeJiiStatPandemi = computeMaeStatistic(weeklyVolatility['JII Volatility'], pandemiMask)
maeJiiStatPost    = computeMaeStatistic(weeklyVolatility['JII Volatility'], postMask)

maeIssiMlPandemi  = computeMaeMl(weeklyVolatility['ISSI Volatility'], pandemiMask)
maeIssiMlPost     = computeMaeMl(weeklyVolatility['ISSI Volatility'], postMask)
maeIssiStatPandemi = computeMaeStatistic(weeklyVolatility['ISSI Volatility'], pandemiMask)
maeIssiStatPost    = computeMaeStatistic(weeklyVolatility['ISSI Volatility'], postMask)

rmseJiiMlPandemi  = computeRmseMl(weeklyVolatility['JII Volatility'], pandemiMask)
rmseJiiMlPost     = computeRmseMl(weeklyVolatility['JII Volatility'], postMask)
rmseIssiMlPandemi = computeRmseMl(weeklyVolatility['ISSI Volatility'], pandemiMask)
rmseIssiMlPost    = computeRmseMl(weeklyVolatility['ISSI Volatility'], postMask)


def summaryTable(volSeries, rmsePandemi, rmsePost, indexName):
    pandemiMask = (volSeries.index >= pandemiStart) & (volSeries.index <= pandemiEnd)
    postMask    = (volSeries.index >= postPandemiStart) & (volSeries.index <= postPandemiEnd)

    data = [
        {
            'Index': indexName,
            'Period': 'Pandemi',
            'Mean Volatility': volSeries.loc[pandemiMask].mean(),
            'Maximum Spike': volSeries.loc[pandemiMask].max(),
            'RMSE (ML)': rmsePandemi
        },
        {
            'Index': indexName,
            'Period': 'Pasca Pandemi',
            'Mean Volatility': volSeries.loc[postMask].mean(),
            'Maximum Spike': volSeries.loc[postMask].max(),
            'RMSE (ML)': rmsePost
        }
    ]
    return pd.DataFrame(data)


summaryJii = summaryTable(weeklyVolatility['JII Volatility'], rmseJiiMlPandemi, rmseJiiMlPost, 'JII')
summaryJii['RMSE (Statistic)'] = [rmseJiiStatPandemi, rmseJiiStatPost]
summaryJii['MAE (ML)'] = [maeJiiMlPandemi, maeJiiMlPost]
summaryJii['MAE (Statistic)'] = [maeJiiStatPandemi, maeJiiStatPost]

summaryIssi = summaryTable(weeklyVolatility['ISSI Volatility'], rmseIssiMlPandemi, rmseIssiMlPost, 'ISSI')
summaryIssi['RMSE (Statistic)'] = [rmseIssiStatPandemi, rmseIssiStatPost]
summaryIssi['MAE (ML)'] = [maeIssiMlPandemi, maeIssiMlPost]
summaryIssi['MAE (Statistic)'] = [maeIssiStatPandemi, maeIssiStatPost]

summaryAll = pd.concat([summaryJii, summaryIssi], ignore_index=True)

print("\n=== Volatility Summary & RMSE + MAE ML vs Statistik ===")
print(summaryAll)




def computeMlMetrics(volSeries, periodMask, trainRatio):
    dfMl = volSeries.loc[periodMask].copy().to_frame(name='vol')
    dfMl['lag1'] = dfMl['vol'].shift(1)
    dfMl.dropna(inplace=True)

    X = dfMl[['lag1']]
    y = dfMl['vol']

    splitIdx = int(len(dfMl) * trainRatio)
    xTrain, xTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
    rfModel.fit(xTrain, yTrain)
    yPred = rfModel.predict(xTest)

    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)

    return rmse, mae

def computeStatMetrics(volSeries, periodMask):
    dfStat = volSeries.loc[periodMask].copy()
    dfLag = dfStat.shift(1)

    validIdx = dfStat.index.intersection(dfLag.dropna().index)
    yTrue = dfStat.loc[validIdx]
    yPred = dfLag.loc[validIdx]

    rmse = np.sqrt(mean_squared_error(yTrue, yPred))
    mae = mean_absolute_error(yTrue, yPred)

    return rmse, mae

splitRatios = {
    "70:30": 0.7,
    "80:20": 0.8,
    "90:10": 0.9
}

results = []

for splitName, ratio in splitRatios.items():
    for indexName in ['JII Volatility', 'ISSI Volatility']:
        volSeries = weeklyVolatility[indexName]

        rmsePandemiMl, maePandemiMl = computeMlMetrics(volSeries, pandemiMask, ratio)
        rmsePostMl, maePostMl = computeMlMetrics(volSeries, postMask, ratio)

        rmsePandemiStat, maePandemiStat = computeStatMetrics(volSeries, pandemiMask)
        rmsePostStat, maePostStat = computeStatMetrics(volSeries, postMask)

        results.append([indexName, "Pandemi", splitName, rmsePandemiMl, maePandemiMl, rmsePandemiStat, maePandemiStat])
        results.append([indexName, "Pasca Pandemi", splitName, rmsePostMl, maePostMl, rmsePostStat, maePostStat])

summaryGrand = pd.DataFrame(results, columns=[
    "index", "period", "split",
    "rmseMl", "maeMl",
    "rmseStat", "maeStat"
])

print("\n=== Grand Summary RMSE & MAE ML vs Statistik ===")
print(summaryGrand)


jiiData = summaryGrand[summaryGrand['index'] == 'JII Volatility']

labels = [
    "70:30\nPandemi", "70:30\nPasca",
    "80:20\nPandemi", "80:20\nPasca",
    "90:10\nPandemi", "90:10\nPasca"
]

rmseMl = [
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pandemi')]['rmseMl'].values[0],
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pandemi')]['rmseMl'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pandemi')]['rmseMl'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
]

rmseStat = [
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pandemi')]['rmseStat'].values[0],
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pandemi')]['rmseStat'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pandemi')]['rmseStat'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(13,6))
plt.bar(x - width/2, rmseMl, width, label='Machine Learning')
plt.bar(x + width/2, rmseStat, width, label='Statistik (Lag-1)')

plt.xticks(x, labels)
plt.title("Perbandingan RMSE Prediksi Volatilitas JII\nBerbagai Split & Periode")
plt.ylabel("RMSE")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

maeMl = [
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
]

maeStat = [
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pandemi')]['maeStat'].values[0],
    jiiData[(jiiData['split']=='70:30') & (jiiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pandemi')]['maeMl'].values[0],
    jiiData[(jiiData['split']=='80:20') & (jiiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pandemi')]['maeStat'].values[0],
    jiiData[(jiiData['split']=='90:10') & (jiiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
]

plt.figure(figsize=(13,6))
plt.bar(x - width/2, maeMl, width, label='Machine Learning')
plt.bar(x + width/2, maeStat, width, label='Statistik (Lag-1)')

plt.xticks(x, labels)
plt.title("Perbandingan MAE Prediksi Volatilitas JII\nBerbagai Split & Periode")
plt.ylabel("MAE")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


issiData = summaryGrand[summaryGrand['index'] == 'ISSI Volatility']

labels = [
    "70:30\nPandemi", "70:30\nPasca",
    "80:20\nPandemi", "80:20\nPasca",
    "90:10\nPandemi", "90:10\nPasca"
]

rmseMl = [
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pandemi')]['rmseMl'].values[0],
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pandemi')]['rmseMl'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pandemi')]['rmseMl'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pasca Pandemi')]['rmseMl'].values[0],
]

rmseStat = [
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pandemi')]['rmseStat'].values[0],
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pandemi')]['rmseStat'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pandemi')]['rmseStat'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pasca Pandemi')]['rmseStat'].values[0],
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(13,6))
plt.bar(x - width/2, rmseMl, width, label='Machine Learning')
plt.bar(x + width/2, rmseStat, width, label='Statistik (Lag-1)')

plt.xticks(x, labels)
plt.title("Perbandingan RMSE Prediksi Volatilitas ISSI\nBerbagai Split & Periode")
plt.ylabel("RMSE")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

maeMl = [
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pandemi')]['maeMl'].values[0],
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pandemi')]['maeMl'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pandemi')]['maeMl'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pasca Pandemi')]['maeMl'].values[0],
]

maeStat = [
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pandemi')]['maeStat'].values[0],
    issiData[(issiData['split']=='70:30') & (issiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pandemi')]['maeStat'].values[0],
    issiData[(issiData['split']=='80:20') & (issiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pandemi')]['maeStat'].values[0],
    issiData[(issiData['split']=='90:10') & (issiData['period']=='Pasca Pandemi')]['maeStat'].values[0],
]

plt.figure(figsize=(13,6))
plt.bar(x - width/2, maeMl, width, label='Machine Learning')
plt.bar(x + width/2, maeStat, width, label='Statistik (Lag-1)')

plt.xticks(x, labels)
plt.title("Perbandingan MAE Prediksi Volatilitas ISSI\nBerbagai Split & Periode")
plt.ylabel("MAE")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
