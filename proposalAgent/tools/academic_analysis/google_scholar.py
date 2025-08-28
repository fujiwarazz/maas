import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)
import httpx
from serpapi import GoogleSearch



@retry(
    retry=(retry_if_exception_type(httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)

def search_google_sholar(query:str):
    """ search google scholar by query, query can be an article, an author

    Args:
        query (str): an article, an author
    """
    params = {
    "engine": "google_scholar",
    "q": query,
    "api_key": "6d2adf2afad17ed350e212b43f22cb6f0a5927ccf5e576004411ad2de0ca9c3a"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    return organic_results
    
def get_author_detail(author_id:str):
    """Get author detail from google scholar"""
    params = {
    "engine": "google_scholar",
    "q": author_id,
    "api_key": "6d2adf2afad17ed350e212b43f22cb6f0a5927ccf5e576004411ad2de0ca9c3a"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    author_results = results["author_results"]
    return author_results


    

if __name__ == "__main__":
    author_name = "杜一"
    author_detail = get_author_detail(author_name)
    print(author_detail)