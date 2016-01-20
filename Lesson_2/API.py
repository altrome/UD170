import json
import requests

def api_get_request(url):
    # In this exercise, you want to call the last.fm API to get a list of the
    # top artists in Spain.
    #
    # Once you've done this, return the name of the number 1 top artist in Spain.
    data=requests.get(url).text
    data=json.loads(data)

    print data['topartists']['artist'][0]['name']

api_get_request('http://ws.audioscrobbler.com/2.0/?method=geo.getTopArtists&country=Spain&api_key=087c378ad8b15c6805f824fe731e2996&format=json')
