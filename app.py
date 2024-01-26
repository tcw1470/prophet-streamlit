import sys, os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

#import utils

from importlib import reload
reload( utils )

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Create a Prophet model instance
model = Prophet()



addresses =[    
    'Nayapara refugee camp, Bangladesh', 
    'Kutupalong refugee camp, Cox Bazar, Bangladesh',
    'Bidibidi Refugee Settlement, Uganda', 
    'Adjumani refugee camp, Uganda', 
    'Ifo, Dadaab, Kenya',    
    ]

addresses0 = [
  'Beach camp, Palestine',
  'Bureij camp, Palestine',
  'Deir El-Balah Camp, Palestine',
  'Jabalia Camp, Palestine',
  'Khan Younis Camp, Palestine',
  'Maghazi camp, Palestine',
  'Nuseirat camp,Palestine',
  'Rafah camp, Palestine'] 

addresses = addresses[-1]

def get_country(lat, lon):
    url = f'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en&zoom=3'
    try:
        result = requests.get(url=url)
        result_json = result.json()
        return result_json['display_name']
    except:
        return None

print('??', get_country(32.782023,35.478867)) # results in Israel

#@utils.st.cache_resource
def get_ESI( NDAYS = 60 ):
  climate_dfs,debug_str = {},{}
  M,siteids,Lx,Ly,Addr=[],[],[],[],[]
  
  for dtype in [29]:
    try:
      climate_dfs[ dtype ]
    except:
      climate_dfs[ dtype ] = {}
  
    for i,ad in enumerate(addresses):
      geolocator = utils.geopy.geocoders.Nominatim(user_agent="3")
      location = geolocator.geocode( ad )    
      print( ad, location.address )
      print( (location.longitude, location.latitude, ))
      lx,ly=location.longitude, location.latitude
      Addr.append( location.address )
      Lx.append(lx)
      Ly.append(ly)
      k = ad[:8]    
      siteids.append(k)
      try:        
        dat, debug_str[dtype,k] = utils.get_climate_data( lx, ly, ndays = NDAYS, dtype=dtype )  
        if utils.DEBUG:
            utils.st.text( 'Shape of T.S.' )
            utils.st.text( dat.shape )
        climate_dfs[ dtype ][k] = dat
        if 0:
            a = dat.copy()        
            filled = a.ffill()
            col = 'raw_value'
            #res = a[[col]].dropna().join( filled.drop( col, axis=1 )) [ dat.columns.to_list() ]) 
        res = utils.np.nanmedian( dat[ 'raw_value'] )
        M.append( res )                  
      except Exception as e:
        print( f'Get climate: {e}')  
        M.append( utils.np.nan )
          
  return climate_dfs, debug_str, utils.pd.DataFrame( dict( SiteID=siteids, Raw_Measure=M, longitude=Lx, latitude=Ly, Addr=Addr) ) 

NDAYS_user = utils.st.number_input( 'Number of historical records used to train forecaster (units=days):', value="min", 
                                   min_value=60.0, max_value =730.0 )
climate_dfs, debug_str, queried_df = get_ESI( NDAYS = NDAYS_user  )  # get time series for all locations

utils.st.header( 'Geo-coordinates of queried sites' )
utils.st.write( '(Longitude, Latitude) of queried sites given name of the site:' )
utils.st.dataframe( queried_df )

for d in climate_dfs.keys():
    for a in climate_dfs[d].keys():                
    
        # Train the model with the prepared data
        df = climate_dfs[d][a][['date', 'raw_value'] ]
        
        model.fit(df)
        
        # Create a dataframe for the future period to be predicted
        future_df = model.make_future_dataframe(periods=10, freq='MS')
        
        # Perform prediction using the model
        forecast = model.predict(future_df)
        fig = model.plot(forecast)
        
        try:
            st.header("Predictive performance visualized")
            st.plotly_chart(fig, use_container_width=True)     
        except:
            pass
        
        try:
            fig2 = model.plot_components(forecast);
            st.header("Weekly and seasonality trends")
            st.plotly_chart(fig2, use_container_width=True)     
        except:
            pass




utils.st.header( 'Time-series of Evaporative Stress Index (ESI)')
utils.st.text( 'High ESI represents high chance of "flash drought"')
for d in climate_dfs.keys():
    for a in climate_dfs[d].keys():                
        #utils.st.dataframe( climate_dfs[d][a] )
        try:
            fig = utils.px.bar( climate_dfs[d][a], 'date','raw_value', title = f'ESI at {a}' )
            fig.show()
            utils.plotly_chart( fig )
        except Exception as e:
            utils.st.text( e)

        utils.st.download_button(
         f"Download ESI measured at {a} to CSV",
         utils.convert_df( climate_dfs[d][a] ),
         f"ESI_last60d_{a}_{utils.date_today}.csv",
         "text/csv",
         key=f'download-esi{d}_{a}'
        )

utils.st.header( 'Debugging info' )
