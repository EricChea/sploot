from geopy import geocoders

def lat_long(query, api_key):
    """
    Parameters
    ----------
    query: str, search query passed to google.
    apikey: str, authentication key obtained through Google console.

    Returns
    -------
    tuple, (latitude, longitude)

    Examples
    --------
    >>> from sploot.credentials_utils import Credentials
    >>> from sploot.geo_utils import lat_long
    >>> creds = Credentials(<PATH_TO_CREDENTIALS.json>)
    >>> # This will depend on how credentials are stored
    >>> apikey = creds.get(["GOOGLE", "MAPS_PLATFORM"])
    >>> lat_long("Mt Everest", apikey)
    (27.9881206, 86.9249751)
    
    Notes
    -----
    2,500 free requests per day, calculated as the sum of client-side and server-side queries.
    50 requests per second, calculated as the sum of client-side and server-side queries.
    """

    g = geocoders.GoogleV3(api_key=api_key)
    geocode = g.geocode(query, timeout=10, language="en")
    return (geocode.latitude, geocode.longitude)
