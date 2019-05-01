'''
************************************************
*Time：2017.9.11       
*Target：All movies' information of IMDB TOP_250
*Resources：http://www.imdb.cn/IMDB250/
************************************************
'''

import re
import requests
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

num = 1 #电影计数
All_txt = [] #全部电影的信息
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'}#浏览器代理
def  getHTMLText(url):
    try:
        #print(url)
        r = requests.get( url,headers = headers )
        #print(r)
        r.encoding = 'utf-8'
        return r.text
    except:
        return "错误"


def get_all_information(url,page):
    '''
    从每一部电影的页面中获取全部信息
    '''
    global num,All_txt
    txt = getHTMLText(url)
    if txt != "错误":
        print('page'+str(page)+' NO.'+str(num)+' Get it!')
    if num == 10:
        print('Finished!!!')
    soup = BeautifulSoup(txt,"html.parser")
    Cname,Ename,Score,title,Actor,Starring,Infor = '','','','','','',''

    #TOP250-film_Chinese_name&Score
    infor_1 = soup.find_all('div',class_ = 'hdd')
    rel = '<h3>'+'[\s\S]*?'+'</h3>'
    pattern = re.compile(rel)
    Cname = ''.join(pattern.findall(str(infor_1[0])))
    Cname = Cname.replace('<h3>','').replace('</h3>','')
    #print(Cname)
    #find_the_year & save
    rel = '（'+'[\s\S]*?'+'）'
    pattern = re.compile(rel)
    time_ = ''.join(pattern.findall(Cname))
    #print(time_)
    with open('time.txt','a',encoding='utf-8') as t:
        t.write( time_.replace('（','').replace('）','') + '\n' )
    #find_Score
    rel = '<i>'+'[\s\S]*?'+'</i>'
    pattern = re.compile(rel)
    Score = ''.join(pattern.findall(str(infor_1[0])))
    Score = Score.replace('<i>','').replace('</i>','')
    #print(Cname,Score)

    #TOP250-film_many_infor
    now = soup.find_all('div',class_ = 'bdd clear')
    #print(now[0])
    a = BeautifulSoup(str(now[0]), "html.parser")
    many_infor = a.find_all('li')

    #TOP250-film_Ename
    Ename = str(many_infor[0]).replace('<li>','').replace('<i>','').replace('</i>','').replace('</li>','').replace('<a>','').replace('</a>','')
    #TOP250-film_Actor
    Actor_temp = BeautifulSoup(str(many_infor[2]), "html.parser").find_all('a')
    Actor = Actor_temp[0].get_text().replace('导演：','')
    #TOP250-film_Starring
    Starring_temp = BeautifulSoup(str(many_infor[3]), "html.parser").find_all('a')
    for i in Starring_temp:
        Starring += i.get_text().replace(' ','') + ' '
    #print(Starring)

    #Top-film_Infor
    for j in range(4,7):
        Infor_temp = BeautifulSoup(str(many_infor[j]), "html.parser")
        for i in Infor_temp.children:
            Infor += i.get_text().replace(' ','') + ' '
        Infor += '\n'
    #print(Infor)

    #TOP250-film_Synopsis
    content =  soup.find_all('div',class_ = 'fk-4 clear')
    #print(content)
    soup_con = BeautifulSoup(str(content[0]), "html.parser")
    title = soup_con.find_all('div',class_ = 'hdd')
    title = str(title[0]).replace('<div class="hdd">','').replace('</div>','\n')
    #print(title)
    content_1 = soup_con.find_all('div',class_ = 'bdd clear')
    content_1 = str(content_1[0]).replace('<div class="bdd clear" style="font-size:15px">','').replace('</div>','')
    content_1 = content_1.replace('<!-- <p><a href="#">更多剧情 >></a></p>  -->','').replace('<br/>','\n')

    #Save_all_information
    All_txt.append('第'+str(num)+'部'+'\n')
    All_txt.append( Cname+'\n' )
    All_txt.append( '【英文名】'+Ename+'\n' )
    All_txt.append( '【评分】'+Score+'\n' )
    All_txt.append( '【导演】'+Actor+'\n' )
    All_txt.append( '【主演】'+Starring+'\n' )
    All_txt.append( Infor+'\n' )
    All_txt.append( title+'\n'+content_1+'\n' )
    All_txt.append('\n')
    num += 1


def getin_one(url,page):
    '''
    在每一页中得到当前页的全部电影的url
    '''
    txt = getHTMLText(url)
    soup = BeautifulSoup(txt, "html.parser")
    #print(soup)
    temp = soup.find_all('div',class_="ss-3 clear")
    rel = '<a href="' + '[\s\S]*?' + '">'
    pattern = re.compile(rel)
    All_url = pattern.findall( str(temp[0]) )
    for i in range(len(All_url)):
        temp_url = 'http://www.imdb.cn'+All_url[i].replace('<a href="','').replace('">','')
        get_all_information(temp_url,page)
    #print(All_url)


def Analyze_some_infor():
    '''
    将所有电影的年份统计并生成条形图
    '''
    plt.rc('font', family='SimHei', size=13)#字体及大小
    #Analyze_time
    file = open('time.txt')
    a,b,c,d,e,f = 0,0,0,0,0,0
    for line in file:
        line = eval(line)
        if line == 0:
            f += 1
        elif line < 1940 and line >= 1920:
            a += 1 
        elif line < 1960 and line >= 1940:
            b += 1
        elif line < 1980 and line >= 1960:
            c += 1
        elif line < 2000 and line >= 1980:
            d += 1
        else:
            e += 1
    times = [a,b,c,d,e,f]
    range_time = ['1920-1940','1940-1960','1960-1980','1980-2000','2000-现在','无信息']
    idx = np.arange(len(range_time))
    width = 0.5
    plt.bar(idx,times,width,color='green')
    plt.xticks(idx+width/2, range_time, rotation=40)
    plt.xlabel('电影年代')
    plt.ylabel('数目')
    plt.savefig('time_pic.png')
    plt.show()

def main():
    global All_txt
    getin_one('http://www.imdb.cn/IMDB250/',1)
    for i in range(2,3):
        getin_one( 'http://www.imdb.cn/imdb250/'+str(i) , i )
    #将已有内容清空
    with open('All_infor.txt','w',encoding='utf-8') as x:
        pass
    with open('All_infor.txt','a',encoding='utf-8') as x:
        for i in All_txt:
            x.write(i)
    Analyze_some_infor()

main()
