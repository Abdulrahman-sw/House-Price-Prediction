from numpy import int64
import pandas as pd

def outliers (df,feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)
    lis = df.index[ (df[feature] < lower) | (df[feature] > upper) ]
    #print("lower:",lower)
    return lis

def remove (df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df
mydf = pd.read_csv("D:\Abdoo\Bachelor's degree\GP\gp prediciton model/finalISA.csv",encoding='ISO-8859-1')
'''
print(df.shape) #berfore preprocessing
df.drop_duplicates(subset=None, keep="first", inplace=True) #removes the duplicates rows
print("after removing duplicates",df.shape)
# Drop odd values in No_rooms
df.drop(df[df["No_rooms"]==' Luxury In Vinci Rest Up To 9  Years."'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' villa in lake view"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='???? ????? E3  ??? ?????  ???? ???? ????"'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' bahary  garden"'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' front of Smouha Club main gate"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='s and Kitchen"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='direct on the sea|Elegant| garage"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='???? ????? C ????? ????? ??? ???? ???? ?????"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='????  ????? j ?????? ???????? ????? ????????????"'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' 150,800 Down Payment"'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' your home in the club & open views"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=='???? ????? C ??? ???????? ???? ???? ??????"'].index, inplace = True) 
df.drop(df[df["No_rooms"]==' 417,480 Down Payment"'].index, inplace = True) 
df.drop(df[df["No_rooms"]=="N/A "].index, inplace = True) 
df.drop(df[df["No_rooms"]=="studio "].index, inplace = True) 

df.loc[df["No_rooms"] == "3 ", "No_rooms"] = "3"
df.loc[df["No_rooms"] == "2 ", "No_rooms"] = "2"
df.loc[df["No_rooms"] == "1 ", "No_rooms"] = "1"
df.loc[df["No_rooms"] == "4 ", "No_rooms"] = "4"
df.loc[df["No_rooms"] == "5 ", "No_rooms"] = "5"
df.loc[df["No_rooms"] == "6 ", "No_rooms"] = "6"
df.loc[df["No_rooms"] == "7 ", "No_rooms"] = "7"
df.loc[df["No_rooms"] == "7+ ", "No_rooms"] = "7"
df = df.dropna(subset=["No_rooms"])
df = df.astype({'No_rooms':'int'})
print("After Num of rooms and baths",df.shape)
#Correcting size
#df['size'] = df['size'].str.replace('sqm','')
#df['size'] = df['size'].str.replace(',','')
#df = df.astype({'size':'int'})

#Correcting Price
#df['price'] = df['price'].str.replace(',','')
#df.drop(df[df["price"]=="Ask for price"].index, inplace = True) 
#df = df.dropna(subset=["price"])

#df = df.astype({'price':'int'})
#df = df.astype({'latitudes':'float64'})

print("after removing odds", df.shape)

#Remove Outliers in size and price

index_lis = []
for feature in ["size","price"]:
    index_lis.extend(outliers(df,feature))
df = remove(df,index_lis)
print("after outliers", df.shape)
#Replace Apartment and Villa
#df['Type'] = df['Type'].str.replace('Apartment','1')
#df['Type'] = df['Type'].str.replace('Villa','2')
#df = df.astype({'Type':'int'})

print("final", df.shape)
print(df.info())

corr_matrix = mydf.corr()
print(corr_matrix)
i=0
j=500
dflist= []
x=1
while (x<=140):
    mydf.sort_values(by = 'size')
    while (i<14800):
        df1 = mydf.iloc[i:j]
        index_lis = []
        for feature in ["size","price"]:
            index_lis.extend(outliers(df1,feature))
        df1 = remove(df1,index_lis)
        #print(x,"  ",df1.shape)
        dflist.append(df1)
        i = j
        j = j + 500
    mydf = pd.concat(dflist)
    x = x + 1
    i= 0
    j = 100
    dflist = []
    print(mydf.shape,"//////////////////////////")
mydf.sort_values(by = ['size'])

mydf.to_csv("D:\Abdoo\Bachelor's degree\GP\gp prediciton model/finalISA.csv", index = False)
corr_matrix = mydf.corr()
print(corr_matrix)
'''
import seaborn as sns

import matplotlib.pyplot as plt
print(mydf["price"])
mydf.plot(x='size', y='price', style='o')
plt.show()

