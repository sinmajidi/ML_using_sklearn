import pandas as pd


data={
    "temp":[35,33,31,29,23,27,34,35,19,3],
    "humidity":[45,55,61,32,56,78,24,56,65,44]
}
df=pd.DataFrame(data)
df.to_csv('output.csv', index=False)
print(df)



read_csv=pd.read_csv('output.csv')
print(read_csv)



print(read_csv.head())
print(read_csv.describe())
print(df['temp'])
print(read_csv.iloc[2])
print(read_csv.iloc[2,1])



filtered_df=read_csv[read_csv['temp']>30]
print(filtered_df)

sorted_df=read_csv.sort_values('temp',ascending=False)
print(sorted_df)

df_grouped=read_csv.groupby('temp')
print(df_grouped.count())



read_csv['wind_speed']=[1,2.3,4,3.3,5,2,1.5,2.7,3.2,4]
print(read_csv)
read_csv=read_csv.drop('wind_speed',axis=1)
print(read_csv)

read_csv.loc[len(read_csv)]={
    "temp":33,
    "humidity":18
}
print(read_csv)