import datetime 
import json
import feedparser
from urllib.request import Request, urlopen
import time
from bs4 import BeautifulSoup

sel_rss={
    'Reuters': ['title','link','published','published_parsed','summary','tags'],
    'BBC': ['title', 'link','published', 'published_parsed','summary'],
    'Investgate': ['title', 'link','published', 'published_parsed', 'summary', 
                   'investegate_headline', 'investegate_company', 
                   'investegate_companycode', 'investegate_companylink', 'investegate_datetime', 
                   'investegate_time', 'investegate_supplier', 'investegate_suppliercode']
        }


rss_links={'Reuters':
           {
            'Business News': 'http://feeds.reuters.com/reuters/businessNews',
            'Company News':'http://feeds.reuters.com/reuters/companyNews',
            'Wealth':'http://feeds.reuters.com/news/wealth',
            'People':'http://feeds.reuters.com/reuters/peopleNews',
            'Politics':'http://feeds.reuters.com/Reuters/PoliticsNews',
            'Science':'http://feeds.reuters.com/reuters/scienceNews',
            'Sports':'http://feeds.reuters.com/reuters/sportsNews',
            'Technology':'http://feeds.reuters.com/reuters/technologyNews',
            'Top News':'http://feeds.reuters.com/reuters/topNews',
            'US':'http://feeds.reuters.com/Reuters/domesticNews',
            'World':'http://feeds.reuters.com/Reuters/worldNews'
           },
          'BBC':
           {
               'Top News':'http://feeds.bbci.co.uk/news/rss.xml',
               'World':'http://feeds.bbci.co.uk/news/world/rss.xml',
               'UK':'http://feeds.bbci.co.uk/news/uk/rss.xml',
               'Asia':'http://feeds.bbci.co.uk/news/world/asia/rss.xml',
               'Europe':'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
               'USnCanada':'http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml',
               'Business':'http://feeds.bbci.co.uk/news/business/rss.xml',
               'UKPolitics':'http://feeds.bbci.co.uk/news/politics/rss.xml',
               'Science':'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
               'Technology':'http://feeds.bbci.co.uk/news/technology/rss.xml'
           },
           'Investgate':
           {
               'General':'https://www.investegate.co.uk/Rss.aspx?type=0'
           }
          }




def parse_news_url(link,source=None):

    try:
        resp = urlopen(Request(link))
        resp_bytes = resp.read()
        exSoup=BeautifulSoup(resp_bytes, "lxml")
    except:
        1
    res={}
    if source=='Reuters':
        
        try:
            read_json=json.loads(exSoup.select('script[type="application/ld+json"]')[0].getText()
                                             .replace('\n','')
                                             .replace('  ','')
                                             .replace("'", "\""))

            try:
                res['Author']=read_json['creator'][0]
            except:
                res['Author']=read_json['creator']
            res['Tags']=read_json['keywords']
        except:
            res['Author']=''
            res['Tags']=''
        try:
            res['Title']=exSoup.select('h1[class="ArticleHeader_headline"]')[0].getText()
        except:
            res['Title']=''

        try:
            res['Content']=exSoup.select('div[class="StandardArticleBody_body"]')[0].getText()
        except:
            res['Content']=''
            
            
    elif source=='BBC':
        
        try:
            res['Title']=exSoup.select('title')[0].getText()
        except:
            res['Title']=''

        try:
            res['Author']=exSoup.select('span[class="byline__name"]')[0].getText().replace('By ','')
        except:
            res['Author']=''
        
        try:
            body_ext=list(set(exSoup.select('p'))-set(exSoup.select('p[class]')))
            body_ext=exSoup.select('p[class="story-body__introduction"]')+body_ext
            body_txt=' '.join([body_ext[i].getText() for i in range(len(body_ext))])
            res['Content']=body_txt.replace('  ','')
        except:
            res['Content']=''
        
        
        try:
            tags=exSoup.select('li[class="tags-list__tags"]')
            res['Tags']=[tags[i].getText() for i in range(len(tags))]
        except:
            res['Tags']=''
    return res

def rss_crawl():
    nws_list=[]
    
    with open('rss_id_list.txt') as json_file:  
        id_list = json.load(json_file)['key']

    dat=str(datetime.datetime.now())
    print('Begun: ' + dat)    
    #id_list=[]

    for srs in rss_links.keys():
        nr_new=0
        nr_repeat=0
        for cats in rss_links[srs].keys():
            NewsFeed = feedparser.parse(rss_links[srs][cats])
            for entry in NewsFeed.entries:
                if entry.get('id','') not in id_list:
                    #print(entry.get('title',''))
                    out={
                        'id': entry.get('id',''),
                        'source': srs,
                        'category': cats,
                        'rss': { sel_key: entry.get(sel_key,'') for sel_key in sel_rss[srs] },
                        'url': parse_news_url(entry.get('link',''),srs)
                        }
                    nws_list=nws_list + [out]
                    id_list=id_list + [entry.get('id','')]
                    nr_new+=1
                else:
                    #print('Repeat')
                    nr_repeat+=1
            #print(cats + ' Done. New: '+ str(nr_new) + ', Repeating: ' + str(nr_repeat))
        dat=str(datetime.datetime.now())    
        print(srs + ' Done. '+dat+' New: '+ str(nr_new) + ', Repeating: ' + str(nr_repeat))


    nws_list_2={'key':nws_list}

    dat=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())).replace(':','-')
    print('Ended: ' + dat)

    with open('rss_crawl' + str(dat) + '.txt', 'w') as f:
        json.dump(nws_list_2, f, ensure_ascii=False)  #194.36.110.236 at 17:50

    id_list_2={'key':id_list}

    with open('rss_id_list.txt', 'w') as f:
        json.dump(id_list_2, f, ensure_ascii=False)
