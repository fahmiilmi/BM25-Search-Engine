import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import math
import numpy as np
import string
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import colorama
from colorama import init, Fore, Back, Style


df = pd.read_excel(r'D:\skripsi2\Manualisasi\New folder\testing\Rekapdata.xlsx', header=None)
df.columns = ["id", "judul", "kategori", "prodi", "katakunci", "abstrak", "tahun"]
datasets = [df.judul.head(700)]
# print(df)
df.to_excel("output.xlsx")



judul = pd.DataFrame(df.judul.head(700))

keyword = pd.DataFrame(df.katakunci.head(700))

datasets =  judul.judul.tolist()
keywords = keyword.katakunci.tolist()
datasets =  judul.judul.tolist()
datasets2 = df


def casefolding(data):
    result = []
    for x in data:
        x = x.lower()
        result.append(x)
    return result
lower = casefolding(datasets)    
casefolding(datasets)



def cleaning(data):
    result = []
    for x in data:
        res = re.sub('#[A-Za-z0-9_]+', " ", x)
        res = re.sub('[^A-Za-z]+', " ", res)
        res = res.strip()
        result.append(res)
    return result
clean = cleaning(lower)
cleaning(lower)


def stopword(data):   
    factory = StopWordRemoverFactory(). get_stop_words()
    more_stopword = ['dan', 'of', 'the','tahun','a'] #menambahkan stopword
    stopw = factory + more_stopword
    dictionary = ArrayDictionary(stopw)
    sto = StopWordRemover(dictionary)
    
    result = []
    for x in range(len(data)):
        stop = sto.remove(data[x])
        result.append(stop)
    return result

stopword(clean)
stop = stopword(clean)
stop2 = stopword(stop)



def tokenize(data):
    result = []
    for x in range(len(data)):
        x = nltk.tokenize.word_tokenize(data[x])
        result.append(x)
    return result
token = tokenize(stop2)
tokenize(stop2)
    


lower2 = casefolding(keywords)    
clean2 = cleaning(lower2)
stop3 = stopword(clean2)
token2=tokenize(stop3)    

tokensearch = [x+y for x,y in zip(token, token2)]


def tf(data):
    term = []
    for x in data:
        for y in x:
            term.append(y)
    term = np.unique(term)

    kolom = [i + 1 for i in range(len(data))]
    
    TF_dict = pd.DataFrame(0, index=term, columns=kolom)

    for x,y in enumerate(data):
        for term in y:
            TF_dict.loc[term, x+1] += 1   
    return TF_dict

TF_dict = tf(tokensearch)
#print(TF_dict)

def df(data):
    
    result = pd.DataFrame(index = data.index, columns = ['df'])
    for term, row in data.iterrows():
        result['df'][term] = sum(1 for x in row if x>0)
    return result

df(TF_dict)
df1 = df(TF_dict)



def dftf(tf, df):
    frames = [tf, df]

    dftf = pd.concat(frames, axis=1)
    dff = dftf.df.tolist()
    return dftf

dftf(TF_dict, df1)
dftf1 = dftf(TF_dict, df1)



def idf(data,df):

    idfDict = pd.DataFrame(index = data.index, columns = ['idf'])
    N = len(data.columns)-1
    
    for term, row in data.iterrows():
        idfDict ['idf'][term]= math.log10((N - data.df[term] + 0.5) /  (data.df[term]+0.5))
        idfDict.append(idfDict) 
    #print(idfDict)
    idfDict = pd.DataFrame(idfDict)
    idfDict = pd.concat([data,idfDict], axis=1)
    return(idfDict)

idfs = idf(dftf1,df1)



def avgdl (data):
    avgl=[]
    for x in data:
        avg = data[x].sum()

        avgl.append(avg)
    
    result = sum(avgl)/len(avgl)
    return result

avgdl(TF_dict)
avgdl1 = avgdl(TF_dict)




def Doclen(data):
    D=[]
    for doc in data:
        avg = data[doc].sum()
        D.append(avg)
    return D

Doclen(TF_dict)
Dlen= Doclen(TF_dict)


def BM25 (data,avg,D):
     
    print("Masukkan kueri:")
    kueri = input()
    kueri = kueri.split()

    try:


        hasilBM = []
        for x in (x for x in kueri if x not in data.index):
            kueri.remove(x)
        for x in (x for x in kueri if x in data.index):
            an = data.loc[x]
            for x in range(len(tokensearch)):
                BM = (an.loc['idf']*((an.loc[x+1]*(1.2+1))/(an.loc[x+1]+1.2*(1-0.75+0.75*D[x]/(avg)))))
                hasilBM.append(BM)
        
        splits = np.array_split(hasilBM, len(kueri))
        split = np.array(splits)
        split = split.T
        split = split.tolist()
        akhir = []
        for x in range(len(split)):
            y = sum(split[x])
            akhir.append(y)
        
        nilai = pd.DataFrame (akhir, columns = ['BM25'])
        index = pd.Index(range(1, 701, 1))
        nilai = nilai.set_index(index)
        mask = nilai['BM25'] != 0
        dfakhir = nilai[mask].sort_values('BM25', ascending=False)

        print(dfakhir)
    except:
        print("Kueri tidak ditemukan di dalam database")    

    return dfakhir



bm25=BM25(idfs,avgdl1,Dlen)


def getfacet(data,value):
    kk= data.katakunci.tolist()    
    facet=[]
    for x in value.index:
        y = kk[x-1]
        z = re.sub(r'\s*,\s*', ',', y) #menghilangkan spasi setelah koma
        facet.append(z)
    return facet

facet=getfacet(datasets2,bm25)



def tokenfacet(data):
    facets = []
    for x in data:
        y = x.strip()
        y=y.replace('\xa0', ' ')  #\xa0 is actually non-breaking space in Latin1 (ISO 8859-1), replace with a space. to utf-8
        chunks = y.split(',')
        z=stopword(chunks)
        z=casefolding(z)
        facets.append(z)
    return facets

facets= tokenfacet(facet)





def showfacet(data):
    term = []
    for x in data:
        for y in x:
            y = y.strip()
            term.append(y)
    sumfacet = {}
    for x in term:
        if x not in sumfacet:
            sumfacet[x] = 0 
        sumfacet[x] += 1
    sort_facet = sorted(sumfacet.items(), key=lambda x: x[1], reverse=True)
    return sort_facet

sort_facet1=showfacet(facets)



# print(facets)
def facetvalue(data):
    dfacet= {}
    for n, i in enumerate (facets):
        for y in i:
            if y in dfacet:
               dfacet[y].extend([n])
            if y not in dfacet:
               dfacet[y] = [n]
    return dfacet

dfacet1=facetvalue(facets)
  
    

    
def pickfacet(facetlist, facet):
    print("Pilih facet:")
    facet_i = int(input())
# facet_i = facet_i -1

    print(facetlist[facet_i][0])
    counter= facet[sort_facet1[facet_i][0]]    
# print(dfacet[sort_facet[facet_i][0]])
    
    for x in counter:
        print('{}. {}'.format(bm25.index[x],datasets2.judul[bm25.index[x]-1],end='\n\n'))
    return

#facet()
# pickfacet(sort_facet1,dfacet1)

def showall(data):     
    y=0
    init(autoreset = True)
    for x in data.index:
    #for y in x:
        print('{}. {}\n>>>Keyword: {}\n'.format(datasets2.id[x-1],datasets2.judul[x-1],Fore.RED+facet[y],end='\n\n'))
        y+=1
    return

# Fore.GREEN+
#showall(bm25)

def menu():
    print("facet filter:")
    print("1. tampilkan semua")
    print("2. pilih facet")
    print("3. Keluar")
    print("Anda ingin filter berdasarkan apa (pilih nomor 1 s.d. 3)?")


    pilih = int(input())
    if pilih == 1:
      showall(bm25)
      menu()
    elif pilih == 2:
      for i in sort_facet1:
      	print(i[0],"(%d)" %i[1])
      pickfacet(sort_facet1,dfacet1)
      menu()       
    elif pilih == 3:
      print("Terima Kasih")  
    return 

menu()
  
