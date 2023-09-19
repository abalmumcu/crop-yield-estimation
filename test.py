import pandas as pd

def reshape_dataframe(df):
    # 'Date' sütununu indeks olarak ayarlayın
    #df.set_index("Date", inplace=True)

    # Aylık verileri ayrı ayrı sütunlara dönüştürün
    df = df.pivot(columns="Date").reset_index()

    # Sütun başlıklarını yeniden düzenleyin
    df.columns = [f"{col[0]}_{col[1]}" if col[0] != "Feature" else col[0] for col in df.columns]

    return df

# Örnek veri çerçevesini oluşturun
data = {
    "Date": ["04-15", "05-15", "06-15"],
    "Feature": [2252, 3343, 4412]
}

df = pd.DataFrame(data)
print(df)
# Veriyi yeniden düzenleyin
reshaped_df = reshape_dataframe(df)

# Yeniden düzenlenmiş veriyi gösterin
print(reshaped_df)
 
months = ["April","May","June","July","August","September"]

df.iloc[row,col]


