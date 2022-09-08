# pip install wikipedia

# import necessary libraries
import wikipedia as wiki  
import pandas as pd
import json

def wiki_molecule(query):
    '''
    function to scrape wikipedia information
    query can be CAS number or molecule name
    '''
    results = wiki.search(query)
    if len(results) == 0:
        return None
    
    page = wiki.page(results[0]) 

    data = {}

    data['page_title'] = page.title
    data['page_url'] = page.url
    data['page_summary'] = page.summary
    data['images'] = page.images



    try:
        tables = pd.read_html(page.url) # read html table
        df1 = pd.DataFrame(tables[0]) # convert to dataframe
        
        data['Compounds_table'] = {k: v.iloc[0, 1].split('  ') for k, v in df1.groupby(0)} # write table as dictionary


    except:
        pass

    data['References'] = page.references

    # convert data dictionary to json 
    #with open("wiki_output.json", "w") as outfile:
    #    file = json.dump(data, outfile)

    #return data
    return data
