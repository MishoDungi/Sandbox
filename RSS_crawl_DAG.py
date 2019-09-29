import datetime 
import json
import feedparser
from urllib.request import Request, urlopen
import time
from bs4 import BeautifulSoup

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator



### ----- ENTITY SOURCING ----- ###

### ----- ELASTICSEARCH ----- ###
from elasticsearch import Elasticsearch, helpers

from NER.ES_NER_Utils import run_ES_art_to_ent

es_link='https://search-news-collection-zixpnvstipu7xweou5ixy4tpse.us-east-2.es.amazonaws.com/'
es=Elasticsearch(es_link)
es_news_log='news-log'
es_news_articles='news-articles'
es_news_entities='news-entities'
es_state='live'
publisher = ['BBC', 'Reuters']
method='elasticsearch'

sel_rss={
    'Reuters': ['title','link','published','published_parsed','summary','tags'],
    'BBC': ['title', 'link','published', 'published_parsed','summary'],
    'Investgate': ['title', 'link','published', 'published_parsed', 'summary', 
                   'investegate_headline', 'investegate_company', 
                   'investegate_companycode', 'investegate_companylink', 'investegate_datetime', 
                   'investegate_time', 'investegate_supplier', 'investegate_suppliercode']
        }

get_sites=['Reuters','BBC']

path='/Users/MD/Downloads/_SCR_REUTERS/RSS_IDs/'

rss_links={'Reuters':
           {
            'Business News': 'http://feeds.reuters.com/reuters/businessNews',
            'Company News':'http://feeds.reuters.com/reuters/companyNews',
            'Wealth':'http://feeds.reuters.com/news/wealth',
            #'People':'http://feeds.reuters.com/reuters/peopleNews',
            #'Politics':'http://feeds.reuters.com/Reuters/PoliticsNews',
            #'Science':'http://feeds.reuters.com/reuters/scienceNews',
            #'Sports':'http://feeds.reuters.com/reuters/sportsNews',
            #'Technology':'http://feeds.reuters.com/reuters/technologyNews',
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
               #'Science':'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
               #'Technology':'http://feeds.bbci.co.uk/news/technology/rss.xml'
           },
           'Investgate':
           {
               'General':'https://www.investegate.co.uk/Rss.aspx?type=0'
           }
          }

def parse_news_url(link,source=None):
    if source in get_sites:
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
                
            try:
                res['Modified']=read_json['dateModified']
            except:
                res['Modified']=''
            try:
                res['Published']=read_json['datePublished']
            except:
                res['Published']=''
            
            res['Tags']=read_json['keywords']
        except:
            res['Author']=''
            res['Tags']=''
        try:
            res['Title']=exSoup.select('h1[class="ArticleHeader_headline"]')[0].getText()
        except:
            res['Title']=''

        try:
            prep_out=exSoup.select('div[class="StandardArticleBody_body"]')[0].select('p')
            res['Content']='\n'.join([par.getText() for par in prep_out])
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
            body_txt='\n'.join([body_ext[i].getText() for i in range(len(body_ext))])
            res['Content']=body_txt.replace('  ','')
        except:
            res['Content']=''
        
        
        try:
            tags=exSoup.select('li[class="tags-list__tags"]')
            res['Tags']=[tags[i].getText() for i in range(len(tags))]
        except:
            res['Tags']=''
    return res

def write_to_log(path,msg):
    with open(path+'log.txt', 'a+', encoding='utf8') as f:
        now = datetime.datetime.now()
        t = now.strftime("%Y-%m-%d %H:%M")
        f.write(msg+str(t) + '\n')


def agg_srs(srs):
    srs_sum={}
    for each in list(set(srs)):
        srs_sum[each]=srs.count(each)
    return srs_sum

def log_new_run(run_name,step,started,finished,stats):
    
    payload={'run':run_name,
             'status':step,
             'run_date':started.replace('-','')[:8],
             'history':[step],
             step+'_note':json.dumps(stats),
             step+'_started':started,
             step+'_complete':finished, 
             
            }
    return payload

def load_art_es(art,st,start_dt,state='test'):
    srs=[]
    tot_len=0    
    if len(art['key'])>0:

        # CREATE ARTICLE RECORDS IN ES
        for idx in range(len(art['key'])):
            if state=='live':
                es_body=art['key'][idx]
                es_body['es_run']=st
                es_body['es_doc']=st+'d'+str(idx)
                es.index(es_news_articles,id=st+'d'+str(idx),body=es_body)
            else:
                print(es_news_articles,st+'d'+str(idx))
            srs=srs+[art['key'][idx]['source']]
            tot_len=tot_len+len(art['key'][idx]['url'].get('Content',''))

        # CREATE NOTES
        notes={'sources':agg_srs(srs),
           'avg_len':int(tot_len/max(len(art['key'])-srs.count('Investgate'),1)),
           'nr_docs':len(art['key']),}

        finish_dt=str(datetime.datetime.now())

        # LOG EVENTS
        payload=log_new_run(st,'ART',start_dt,finish_dt,notes, )
        if state=='live':
            es.index(es_news_log,id=st,body=payload)
        else:
            print(es_news_log,st,payload)
    else:
        notes={}
    return st,notes

def es_msearch_RSS(NewsFeed,es,es_news_articles):
    feed_entries=[entry.get('id','') for entry in NewsFeed.entries]
    srch=[]
    for entry in feed_entries:
        srch.append({'index':es_news_articles})
        srch.append({'query': {'match': {'id' : entry }}})
    res_srch=es.msearch(body=srch,index=es_news_articles,filter_path='responses.hits.hits._source.id')
    res_srch2=[[hit2['_source']['id'] for hit2 in hit1['hits']['hits']] for hit1 in res_srch['responses']]
    res_srch3=[item for sublist in res_srch2 for item in sublist]
    return res_srch3


def rss_crawl():
    nws_list=[]

    start_dt=str(datetime.datetime.now())
    st=start_dt.replace(' ','_').replace('-','').replace(':','')[:13]

    if method=='local':
        with open(path+'rss_id_list.txt') as json_file:  
            id_list = json.load(json_file)['key']
    else:
        id_list=[]

    write_to_log(path,'Begun: ') if method=='local' else ''
    #id_list=[]

    for srs in rss_links.keys():
        nr_new=0
        nr_repeat=0
        for cats in rss_links[srs].keys():
            NewsFeed = feedparser.parse(rss_links[srs][cats])
            if method=='elasticsearch':
                best_hits=es_msearch_RSS(NewsFeed,es,es_news_articles)
            for entry in NewsFeed.entries:
                if ((entry.get('id','') not in id_list) and (method=='local')) or \
                ((entry.get('id','') not in best_hits) and (method=='elasticsearch')):
                    print(entry.get('id','')) if es_state!='live' else ''
                    out={
                        'id': entry.get('id',''),
                        'source': srs,
                        'category': cats,
                        'rss': { sel_key: entry.get(sel_key,'') for sel_key in sel_rss[srs] },
                        'url': parse_news_url(entry.get('link',''),srs)
                        }
                    nws_list=nws_list + [out]
                    if method=='local':
                        id_list=id_list + [entry.get('id','')]

                    nr_new+=1
                else:
                    #print('Repeat')
                    nr_repeat+=1
        print(srs + ' Done. New: '+ str(nr_new) + ', Repeating: ' + str(nr_repeat))
        write_to_log(path,srs + ' Done. New: '+ str(nr_new) + ', Repeating: ' + str(nr_repeat) + ' , Time: ') \
            if method=='local' else ''

    if len(nws_list)>0:

        art_collection={'key':nws_list}
        if method=='local':
            #dat=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())).replace(':','-')
            with open(path+'rss_crawl' + st + '.txt', 'w') as f:
                json.dump(art_collection, f, ensure_ascii=False)  #194.36.110.236 at 17:50

            id_list_2={'key':id_list}

            with open(path+'rss_id_list.txt', 'w') as f:
                json.dump(id_list_2, f, ensure_ascii=False)
        elif method=='elasticsearch':
            st,notes=load_art_es(art_collection,st,start_dt,state=es_state)

        write_to_log(path,'Ended: ') if method=='local' else ''
    else:
        write_to_log(path,'Nothing to say. Ended: ') if method=='local' else ''

def run_ES_instance():
    return run_ES_art_to_ent(es,  es_news_log,  es_news_articles,  es_news_entities, publisher, 3)
def simple_def():
    import os
    print(os.environ)
    return os.environ
  

default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2019, 9, 20, 22, 00, 00),
    'concurrency': 1,
    'retries': 0
}

with DAG('RSS_CRWL',
         default_args=default_args,
         schedule_interval='*/10 * * * *',
         ) as dag:
    opr_open = BashOperator(task_id='say_Hi',
                             bash_command='echo "Hi!!"')
        
    opr_greet = PythonOperator(task_id='greet',
                            python_callable=rss_crawl)
    opr_sleep = PythonOperator(task_id='sleep_me',
                    python_callable=run_ES_instance)

opr_greet >> opr_sleep
