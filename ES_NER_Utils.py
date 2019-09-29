import os
import json
import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_file_kim.json"
#with open('key_file_kim.json','r') as f:
#    file=json.load(f)
    
from google.cloud import language
language_client = language.LanguageServiceClient()

from elasticsearch import Elasticsearch, helpers

### ----- NER TOOLING ----- ###
# Run entity detection.
def detect_entities(txt, language_client):
    document = language.types.Document(
        content=txt,
        type=language.enums.Document.Type.PLAIN_TEXT,
        language="en")
    entities = language_client.analyze_entities(document).entities
    return entities

def article_entity_extract(qry,language_client):
    out=detect_entities(qry, language_client)

    res_ner=[{'name':out[idx].name,
      'type':out[idx].type,
      'salience':out[idx].salience,
      'mid':out[idx].metadata['mid'],
      'url':out[idx].metadata['wikipedia_url'],
      'mentions_unq':list(set([out[idx].mentions[i].text.content for i in range(len(out[idx].mentions))])),
      'mentions_len': len(out[idx].mentions)} 
      for idx in range(len(out))]
    return res_ner

def collection_entity_extract(art_collect, publisher, language_client, verbose=0):
    tot_ents=0
    dt0=datetime.datetime.now()
    for idx in range(len(art_collect)):
        if art_collect[idx]['feed']['publisherName'] in publisher:
            txt_string=str(art_collect[idx]['Title']+ '\n'+str(art_collect[idx]['text']))
            
            assert(type(txt_string) is str)            
            art_collect[idx]['entities']=article_entity_extract(txt_string,language_client)
            tot_ents=tot_ents+len(art_collect[idx]['entities'])
        
    return art_collect,tot_ents


### ----- LOGGING TOOLING ----- ###

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


def find_open_jobs(prev_step,es,max_jobs=3,col_log='news-log',col_arts='news-articles'):
    # Search for any items in the collection with status prev_step
    res_run=es.search(col_log,{'query':{'bool':{'must':
                      [{'match':{'status': prev_step}}]}}}, \
                        filter_path='hits.hits._id',size=max_jobs)
    
    # CREATE LIST OF RUN IDs to process
    if len(res_run)>0:
        res_run_list=[hit['_id'] for hit in res_run['hits']['hits']]
        # UPDATE STATUS TO RUNNING
        [es.update(col_log,run,body={'doc':{'status': 'RUNNING'}}) for run in res_run_list]
    else:
        res_run_list=[]
    return res_run_list

def build_es_art_collect(res_run_list,es,col_arts='news-articles'):
    # BUILD ART_COLLECT
    # GET ALL ARTICLES FROM COLLECTIONS
    qry_res=[es.search(body={'query':{'term':{'es_run':run_id}}},
                      index=col_arts,size=10000)['hits']['hits'] for run_id in res_run_list]
    qry_res=[item for sublist in qry_res for item in sublist]
    
    # EXTRACT ALL ARTICLES FROM RESULTS
    art_collect=[hit['_source'] for hit in qry_res]
    return art_collect


def update_to_done(done_step,res_run_list,start_dt,finish_dt,note,es,col_log='news-log'):
    run_logs=es.mget({'ids':res_run_list},index=col_log)['docs']
    hist_list=[run['_source']['history'] for run in run_logs]
    
    [es.update(col_log,run,body={'doc':{'status': done_step,
                            'history': hist_list[i]+[done_step],
                            done_step+'_started': start_dt,
                            done_step+'_complete': finish_dt,
                            done_step+'_note': note},
                            }) for i,run in enumerate(res_run_list)]
    
    

def rename_keys(js1):
    js=js1.copy()
    js['site']=js['url']
    repl_key(js,'rss','feed')    
    js['url']=js['feed']['link']
    del js['feed']['link']
    js['feed']['publisherName']=js['source']
    del js['source']
    js['Title']=js['site']['Title']
    del js['site']['Title']
    js['text']=js['site']['Content']
    del js['site']['Content']
    
    dt_rss=js['feed']['published_parsed']
    dt_dt=datetime.datetime(dt_rss[0],dt_rss[1],dt_rss[2],dt_rss[3],dt_rss[4],dt_rss[5],dt_rss[6],)
    dt=int(datetime.datetime.timestamp(dt_dt)*1000)
    js['pub_date']={}
    js['pub_date']['$date']=dt
    return js

def repl_key(js, old, new):
    js[new]=js[old]
    del js[old]
    return js





def run_ES_art_to_ent(es, es_news_log, es_news_articles, es_news_entities, publisher, max_jobs=3, verbose=0):
    start_dt=str(datetime.datetime.now())

    # FIND 3 OPEN JOBS FROM 27-SEP and give me the names of those runs, mark them running
    res_run_list=find_open_jobs('ART',es=es,max_jobs=max_jobs,col_log=es_news_log)
    print('Runs',res_run_list) if verbose>0 else ''
    if len(res_run_list)>0:
        # SEARCH FOR ALL ARTICLES IN THE RUNS SELECTED AND RETURN THEM IN LIST
        art_collect=build_es_art_collect(res_run_list,col_arts=es_news_articles, es=es)
        # CONFORM TO DOWNSTREAM FORMAT
        art_collect=[rename_keys(art_collect[i]) 
                            for i in range(len(art_collect)) 
                            if art_collect[i]['source'] in publisher]
        print('Ready articles') if verbose>0 else ''
        art_ent_collect, tot_ents=collection_entity_extract(art_collect, publisher, language_client)
        print('Ready entities') if verbose>0 else ''
        for idx in range(len(art_ent_collect)):
            es.index(es_news_entities,id=art_ent_collect[idx]['es_doc'],body=art_ent_collect[idx])

        print('Ready ES') if verbose>0 else ''

        finish_dt=str(datetime.datetime.now())
        notes={'batch':res_run_list,
               'nr_docs_batch':len(art_ent_collect),
               "avg_len_batch":tot_ents/max(len(art_ent_collect),1)}
        update_to_done('NER',res_run_list,start_dt,finish_dt,notes,es=es,col_log=es_news_log)
