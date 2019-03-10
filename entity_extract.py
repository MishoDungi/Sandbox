from google.cloud import language
from urllib.parse import quote_plus

from re import compile
from re import IGNORECASE
from requests import get

# The URL for a GET request to the Wikidata API. The string parameter is the
# SPARQL query.
WIKIDATA_QUERY_URL = "https://query.wikidata.org/sparql?query=%s&format=JSON"

# A Wikidata SPARQL query to find stock ticker symbols and other information
# for a company. The string parameter is the Freebase ID of the company. a
MID_TO_TICKER_QUERY = (
    'SELECT ?companyLabel ?rootLabel ?tickerLabel ?exchangeNameLabel'
    ' WHERE {'
    '  ?entity wdt:P646 "%s" .'  # Entity with specified Freebase ID.
    '  ?entity wdt:P176* ?manufacturer .'  # Entity may be product.
    '  ?manufacturer wdt:P1366* ?company .'  # Company may have restructured.
    '  { ?company p:P414 ?exchange } UNION'  # Company traded on exchange ...
    '  { ?company wdt:P127+ / wdt:P1366* ?root .'  # ... or company has owner.
    '    ?root p:P414 ?exchange } UNION'  # Owner traded on exchange or ...
    '  { ?company wdt:P749+ / wdt:P1366* ?root .'  # ... company has parent.
    '    ?root p:P414 ?exchange } .'  # Parent traded on exchange.
    '  VALUES ?exchanges { wd:Q13677 wd:Q82059 } .'  # Whitelist NYSE, NASDAQ.
    '  ?exchange ps:P414 ?exchanges .'  # Stock exchange is whitelisted.
    '  ?exchange pq:P249 ?ticker .'  # Get ticker symbol.
    '  ?exchange ps:P414 ?exchangeName .'  # Get name of exchange.
    '  FILTER NOT EXISTS { ?company wdt:P31 /'
    '                               wdt:P279* wd:Q1616075 } .'  # Blacklist TV.
    '  FILTER NOT EXISTS { ?company wdt:P31 /'
    '                               wdt:P279* wd:Q11032 } .'  # Blacklist news.
    '  SERVICE wikibase:label {'
    '   bd:serviceParam wikibase:language "en" .'  # Use English labels.
    '  }'
    ' } GROUP BY ?companyLabel ?rootLabel ?tickerLabel ?exchangeNameLabel'
    ' ORDER BY ?companyLabel ?rootLabel ?tickerLabel ?exchangeNameLabel')

# Run entity detection.
def detect_entities(txt):
    language_client = language.LanguageServiceClient()
    document = language.types.Document(
        content=txt,
        type=language.enums.Document.Type.PLAIN_TEXT,
        language="en")
    entities = language_client.analyze_entities(document).entities
    return entities


def collect_entities(entities):
    # Collect all entities which are publicly traded companies, i.e.
    # entities which have a known stock ticker symbol.
    companies = []
    for entity in entities:

        # Use the Freebase ID of the entity to find company data. Skip any
        # entity which doesn't have a Freebase ID (unless we find one via
        # the Twitter handle).
        name = entity.name
        metadata = entity.metadata
        try:
            mid = metadata["mid"]
        except KeyError:
            #self.logs.debug("No MID found for entity: %s" % name)
            continue

        company_data = get_company_data(mid)

        # Skip any entity for which we can't find any company data.
        if not company_data:
            #self.logs.debug("No company data found for entity: %s (%s)" %
            #                (name, mid))
            continue
        #self.logs.debug("Found company data: %s" % company_data)

        for company in company_data:

            # Extract and add a sentiment score.
            '''sentiment = self.get_sentiment(text)'''
            #self.logs.debug("Using sentiment for company: %s %s" %
            #                (sentiment, company))
            '''company["sentiment"] = sentiment'''

            # Add the company to the list unless we already have the same
            # ticker.
            tickers = [existing["ticker"] for existing in companies]
            if not company["ticker"] in tickers:
                companies.append(company)
            #else:
            #    self.logs.warn(
            #        "Skipping company with duplicate ticker: %s" % company)
    return companies

    
def make_wikidata_request(query):
    """Makes a request to the Wikidata SPARQL API."""

    query_url = WIKIDATA_QUERY_URL % quote_plus(query)
    #self.logs.debug("Wikidata query: %s" % query_url)
    #print(query_url)
    response = get(query_url)
    try:
        response_json = response.json()
    
    except ValueError:
    #    self.logs.error("Failed to decode JSON response: %s" % response)
        return None
    #self.logs.debug("Wikidata response: %s" % response_json)

    try:
        results = response_json["results"]
        bindings = results["bindings"]
    
    except KeyError:
    #    self.logs.error("Malformed Wikidata response: %s" % response_json)
        return None

    return bindings


def get_company_data(mid):
    """Looks up stock ticker information for a company via its Freebase ID.
    """

    query = MID_TO_TICKER_QUERY % mid
    bindings = make_wikidata_request(query)

    if not bindings:
        #self.logs.debug("No company data found for MID: %s" % mid)
        return None

    # Collect the data from the response.
    datas = []
    for binding in bindings:
        try:
            name = binding["companyLabel"]["value"]
        except KeyError:
            name = None

        try:
            root = binding["rootLabel"]["value"]
        except KeyError:
            root = None

        try:
            ticker = binding["tickerLabel"]["value"]
        except KeyError:
            ticker = None

        try:
            exchange = binding["exchangeNameLabel"]["value"]
        except KeyError:
            exchange = None

        data = {"name": name,
                "ticker": ticker,
                "exchange": exchange}

        # Add the root if there is one.
        if root and root != name:
            data["root"] = root

        # Add to the list unless we already have the same entry.
        if data not in datas:
            #self.logs.debug("Adding company data: %s" % data)
            datas.append(data)
        #else:
        #    #self.logs.warn("Skipping duplicate company data: %s" % data)

    return datas
    
import os
import pandas as pd
from bs4 import BeautifulSoup
import sys

def extract_ft_articles(path, save_file_name):
    df = pd.DataFrame([],columns=['File','Title','Time', 'Author', 'Description', 'Content', 'Snippet', 'Tags', 'Links'])
    ii=0

    for root, dirs, files in os.walk("./" + path):  
        for filename in files:

            #print(path + "/" + filename)
            resp = open(path + "/" + filename, "rb")#urlopen(Request(url))
            resp_bytes = resp.read()
            exSoup=BeautifulSoup(resp_bytes, "lxml")


            try:
                title = exSoup.select('title')[0].getText()
            except:
                title = ''
            try:
                time = exSoup.select('span[class="time"]')[0].getText()
            except:
                time = ''

            try:
                hidden=exSoup.select('input[name="data"]')[0].attrs['value']
                author = json.loads(hidden)['data']['byline']
                try:
                    snip = json.loads(hidden)['data']['snippet']
                except:
                    snip= ''
            except:
                author=''

            try:
                description = exSoup.select('meta[name="description"]')[0].attrs['content']
            except:
                description = ''

            try:
                content = exSoup.select('div[id="storyContent"]')[0].getText()
            except:
                content = ''

            try:
                tags = exSoup.select('div[data-track-comp-name="relatedTopics"]')[0].select('a')[0].getText()
            except:
                tags = ''

            try:
                lnks_raw=exSoup.select('div[id="storyContent"]')[0].select('a')
                links = [[i.getText(),i.attrs['href']] for i in lnks_raw]
            except:
                links = ''

            add = {'File': filename,
                   'Title': title,
                   'Time': time , 
                   'Author':author, 
                   'Description':description, 
                   'Content':content, 
                   'Snippet':snip, 
                   'Tags':tags, 
                   'Links':links}
            df = df.append(add, ignore_index = True)
            ii+=1
            sys.stdout.write('\r'+filename+" Progress {:2.2%}".format(ii / len(files)))
            sys.stdout.flush()
            #print("Progress {:2.2%}".format(ii / len(files)), end="\r")

    df.to_csv(save_file_name+'.csv')

