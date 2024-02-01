import sys, os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

gparent_dir = os.path.dirname( parent_dir )
sys.path.append(gparent_dir)
sys.path.append(parent_dir)
# sys.path.append( parent_dir + 'rcamps' )

print( 'apps.py\n Current',os.curdir, 'parent:', parent_dir, 'Granny:', gparent_dir)

import utils
from importlib import reload
reload( utils )


import streamlit as st 

# ================== header ==================
tit='Refugee Watch @streamlit'
st.set_page_config(
  page_title=tit, 
  page_icon="^-^",     
  initial_sidebar_state='expanded'
)
st.title( tit )
st.write( "Thank you for visiting the Refugee-Watch ^-^" )
st.write( "This is a working prototype being developed to help locate camps currently at high risks of zero water and floods.\nPlease revisit for upgrades.")


intro_markdown = utils.read_markdown_file( utils.Path( "README.md") )
st.markdown(intro_markdown, unsafe_allow_html=True)

 
utils.st.dataframe( utils.pd.read_excel( 'data/_database_digital_rev.4_7_21.xlsx' ) )
