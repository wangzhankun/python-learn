import re
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

def  getHTMLText(url):
    try:
        #print(url)
        r = requests.get(url)
        #print(r)
        r.encoding = 'utf-8'
        return r.text
    except:
        return "错误"


class LectureAgent:
    """
    可将讲座源中的讲座分发到讲座目的地的对象
    """
    def __init__(self):
        self.sources = []
        self.destinations = []

    def add_source(self, source):
        self.sources.append(source)

    def addDestination(self, dest):
        self.destinations.append(dest)

    def distribute(self):
        """
        从所有讲座源中获取所有的讲座。并将其分发到所有的讲座目的地
        """
        items = []
        for source in self.sources:
            items.extend(source.get_items())
        for dest in self.destinations:
            dest.receive_items(items)


class LectureItem:
    """
    有标题和正文组成的简单讲座
    """
    def __init__(self, title, body):
        self.title = title
        self.body = body



class SimpleWebSource:
    """
    使用正则表达式从网页中提取讲座的讲座源
    """
    def __init__(self, url, encoding = 'utf-8'):
        self.url = url
        self.urls = []
        self.encoding = encoding

    def get_items(self):
        for url in self.urls:
            txt = getHTMLText(url)
            soup = BeautifulSoup(txt,"html.parser")
            info = soup.find_all('div',class_ = 'art_title')
            rel = '<h1>'+'[\s\S]*?'+'</h1>'
            pattern = re.compile(rel)
            arc_title = ''.join(pattern.findall(str(info[0])))
            #arc_title = arc_title.replace('<h1>','').replace('</h1>','')
            #print(arc_title)
            info = soup.find_all('div',class_ = 'art_zw')
            rel = '<p>'+'[\s\S]*?'+'</p>'
            pattern = re.compile(rel)
            body = ''
            for i in info:
                para = ''.join(pattern.findall(str(i)))
                body += para
            #body = body.replace('<p>','').replace('</p>','').replace('<span>','')
            yield LectureItem((arc_title.replace('h1', 'h3')), body + '\n')
            #print(arc_title,body)

    def get_urls(self):
        '''
        在每一页中得到当前页的全部讲座的url
        '''
        txt = getHTMLText(self.url)
        soup = BeautifulSoup(txt, "html.parser")
        #print(soup)
        temp = soup.find_all('div',class_="title_rtcon")
        rel = '<a href="' + '[\s\S]*?' + '">'
        pattern = re.compile(rel)
        All_url = pattern.findall( str(temp[0]) )
        for url in All_url:
            temp_url = 'http://www.bit.edu.cn/tzgg17/jzyg2/'+ url.replace('<a href="','').replace('">','')
            self.urls.append(temp_url)
            #print(temp_url)



        

class PlainDestination:
    """
    以纯文本方式显示所有讲座的讲座目的地
    """
    def receive_items(self, items):
        for item in items:
            print(item.title)
            print('-' * len(item.title))
            print(item.body)


class HTMLDestination:
    """
    以HTML格式显示所有讲座的讲座目的地
    """
    def __init__(self, filename):
        self.filename = filename

    def receive_items(self, items):
        out = open(self.filename, 'w')
        print("""
        <html>
            <head>
                <title>Tody's Lecture</title>
            </head>
            <body>
                <h1>Tody's Lecture</h1>
        """,file = out
        )
        print('<ul>', file = out)
        id = 0
        for item in items:
            id += 1
            print('<li><a href="#{}">{}</a></li>'.format(id, item.title), file = out)
            print('</ul>', file = out)

        id = 0
        for item in items:
            id += 1
            print('<h2><a name="{}">{}</a></h2>'.format(id, item.title), file = out)
            print('{}'.format(item.body.replace(u'\xa0', u' ')), file = out)
            #print('<pre>{}</pre>'.format(item.body), file = out)
        print("""
            <body>
                </html>
        """,file = out
        )
        out.close()


def runDefaultSetup():
    """
    默认的讲座预告源
    """
    agent = LectureAgent()
    #从北理工讲座预告获取信息：
    bit_url = 'http://www.bit.edu.cn/tzgg17/jzyg2/index.htm'
    bit = SimpleWebSource(bit_url)
    bit.get_urls()
    agent.add_source(bit)
    #bit.get_items()
    #添加纯文本目的地和HTML目的地
    agent.addDestination(PlainDestination())
    agent.addDestination(HTMLDestination('Lecture.html'))

    #分发讲座
    agent.distribute()
    

if __name__ == '__main__': 
    runDefaultSetup()