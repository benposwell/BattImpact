import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
import base64

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon=":battery:")
nav = get_nav_from_toml(".streamlit/pages.toml")

pg = st.navigation(nav)

pg.run()