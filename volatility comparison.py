import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

#tampilin 5 data
print(weeklyLogReturns.head())



# grafik komparasi log return issi dan jii
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


