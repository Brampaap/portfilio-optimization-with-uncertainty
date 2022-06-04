import streamlit as st
import pandas as pd
import numpy as np
np.random.seed(seed=2)

import random
from scipy.optimize import minimize
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import pickle
import json
import time

# Predicted securities for current month
inference_data = pickle.load(open('inference_data.pickle', 'rb'))

# 2nd step optimizer
class Minimizer():
    def __init__(self, n_securities=0, divercity_ratio=0.3, allocs_init=None):
        self.n_securities = n_securities
        self.divercity_ratio = divercity_ratio
        
        self.bounds_tuples = [(0, self.divercity_ratio)]*self.n_securities
        
        self.constraints = {'type': 'eq', 
                            'fun': lambda x: abs(np.sum(x)-1)}
        
        
        if allocs_init is None:
            self.allocs_init = np.random.uniform(0, 0.01, self.n_securities)
        else: 
            self.allocs_init = allocs_init
            
    def neg_sharpe_ratio(self, allocs, prices, uncertainty):
        """Calculates negative sharpe ratio"""
        
        numerator = (np.sum(allocs*prices)-0.008)
        denumerator = np.mean((allocs > 0)*uncertainty)
        
        return -numerator/denumerator


    def optimize(self, prices: list = [], uncertainty: list = []):
        
        alloc = minimize(fun = self.neg_sharpe_ratio, 
                         x0 = self.allocs_init,
                         args = (prices, uncertainty),
                         bounds = self.bounds_tuples,
                         constraints = self.constraints
                        ).x
        
        alloc = np.round(alloc, 2)
        
        return alloc

def change_text():
    """
    Select random text for user
    """
    
    output_text = ["Отлично! 🎉 Вот, что у меня получилось 📑", "Здорово! Давай посмотрим, что получилось... 🤔", "Хорошая работа, давай посмотрим на результат  🥳", "Спасибо, что выбрали наш сервис! 🥰", "Результат на следующий месяц 🗓", "🎸 Rock&Roll 🤘 и немного магии 🪄, и вот что получилось", "ВЖУУХ! ВЖУУХ! 🎈 и портфель готов", "Би-Бу-БИП 😵‍💫💬📝💵...💼"]

    st.session_state['get_text'] = random.choice(output_text)

#----------------------WEB-APP----------------------

# Welcome part
st.title('Standalone portfolio optimization :unicorn_face:')

st.markdown("<h1 style='text-align: center; color: black;'>Добро пожаловать!</h1>", unsafe_allow_html=True)


st.markdown('Перед Вами демонстрационный стенд, сервиса **оптимизации** :bank: <span style="color: #8f8fef">инвестиционного портфеля</span>.', unsafe_allow_html=True)

# Info
st.info('Представленный сервис способен как ребаллансировать текущий портфель, так и предложить независимую рекомендацию на следующий месяц.')

# Radio
opt_type_value = st.radio(
     "Выберите тип желаемоей действите",
     ('Ребаллансировка текущего портфеля', 
      'Получить независимую рекомендацию'))

if opt_type_value == 'Ребаллансировка текущего портфеля':
    
    st.markdown(f"Вы выбрали: <span style='color:red'>{opt_type_value}</span>", unsafe_allow_html=True)
    st.markdown("1. Введите параметры ваше портфеля") 
    
    # Setup defaul Dataframe in session for continuous update
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["Ticker", 
                                                    "Percentage"])
    
    # Table settings
    gb = GridOptionsBuilder.from_dataframe(st.session_state.df)
    gb.configure_column('Ticker', editable = False)  
    gb.configure_column('Percentage', editable = True) 
    
    go = gb.build()

    # Form that creates new line in table
    with st.form(key='add_security'):
        
        # Columns separate page on N grid-like parts
        col1, col2 = st.columns(2)

        with col1:
            ticker = st.selectbox("Тикер", tuple(sorted([x for x in inference_data.full_name.to_list()])))

        with col2:
            percentage = st.slider("Доля в портфеле", min_value=0.01, max_value=1., step=0.01)
        
        submit_button = st.form_submit_button("Добавить актив")

    # Logic when form submitted
    if submit_button:
        
        result_info = st.empty()
        
        if ticker not in st.session_state.df['Ticker'].values:
            st.session_state.df = st.session_state.df.append(pd.DataFrame({"Ticker":[ticker], \
                                                                           "Percentage":[str(percentage)]}), 
                                                             ignore_index=True)
       
    else: 
            
            st.session_state.df.loc[st.session_state.df['Ticker']==ticker, 'Percentage'] = str(percentage)
    
    # Show table
    grid_return = AgGrid(st.session_state.df, 
                         gridOptions=go, 
                         theme='streamlit', 
                         fit_columns_on_grid_load=True, 
                         height=200, 
                         update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
                         custom_css={".ag-row-hover, .ag-active-item": {
                                        "color": "white !important"},
                                     ".ag-cell-inline-editing": {
                                        "color": "black !important"}
                                    })
    
    # When "" or NaN table'll remove this row.
    without_na = grid_return['data'][grid_return['data']!=''].dropna()
    
    if without_na.shape[0] != st.session_state.df.shape[0] :
        st.session_state.df = without_na
        st.experimental_rerun()
        
    cols = st.columns(2)
    
    # Other users setting
    with cols[0]:
        
        d = st.slider("2. Укажите максимальную долю на один актив", 
                      min_value=0.1, 
                      max_value=1., 
                      step=0.1)
    
    with cols[1]:
        
        threshold = st.select_slider("3. Выберите ваш риск-профиль", 
                                     ['Консервативный 🐣', 'Умеренный 🙊', 'Агрессивный 🦅'])
        
        threshold = {"Консервативный":0.001299, 
                     "Умеренный":0.002591, 
                     "Агрессивный":0.005104}[threshold[:-2]]
        
    # <br>
    st.markdown('')
    
    # Optimize button
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown('---')
    with cols[2]:
        st.markdown('---')
    
    with cols[1]:
        opt_but = st.empty()
        optimize = opt_but.button('Оптимизировать портфель', on_click=change_text)
    result_info = st.empty()
    
    # Runs loader when if necessary
    with st.spinner('Минуточку. Считаю-считаю...'):
        
        if optimize:
            
            # Handle empty df
            if st.session_state.df.shape[0]==0:
                result_info.error('Ваш портфель пуст, перейдите на шаг 1!')

            else:
                # Users allocations
                allocs_init = st.session_state.df.Percentage.to_numpy().astype(np.float32)
                
                # Number of securities
                n_securities = st.session_state.df.shape[0]
                
                # Takes slice of full available data
                slice_of_data = inference_data[inference_data.full_name\
                                                .isin(st.session_state.df.Ticker)]
                
                prices = slice_of_data['y_pred'].to_numpy()
                
                
                uncertainty = slice_of_data['total_var'].to_numpy()
                
                # Risk-profile cut
                uncertainty = np.where(uncertainty>threshold, 1, uncertainty)
                
                # Get interested allocations
                allocs_creator = Minimizer(n_securities=n_securities, 
                                           divercity_ratio=d, 
                                           allocs_init=allocs_init)

                allocs = allocs_creator.optimize(prices, uncertainty)

                # Prettify the output table
                slice_of_data.loc[:, 'Percentage'] = allocs
                slice_of_data.rename(columns={"full_name":"Ticker"}, inplace=True)

                # Result table
                st.markdown(f"<h2>{st.session_state.get_text}</h2>", unsafe_allow_html=True)
                
                st.table(slice_of_data.loc[allocs>0, ["Ticker", "Percentage"]].reset_index(drop=True))

                # Metrics
                default_return = sum(prices*allocs_init)*100
                expected = sum(slice_of_data.Percentage*slice_of_data['y_pred'])*100
                
                cols = st.columns(2)
    
                with cols[0]:
                    st.metric(label="Доходность вашего портфеля",
                              value="".join([str(round(default_return, 2)),"%"]))
                
                with cols[1]:
                    delta = "".join([str(round((expected-default_return), 2)),"%"]) 
                    st.metric(label="Доходность ребаллансированного портфеля",
                              value="".join([str(round(expected, 2)), "%"]), 
                              delta=delta)
            
else:
    
    st.markdown(f"Вы выбрали: <span style='color:red'>{opt_type_value}</span>", unsafe_allow_html=True)
    
    # Setup default session df
    if "dff" not in st.session_state:
        st.session_state.dff = pd.DataFrame()
    
    # Init result df
    final_rec = pd.DataFrame()
    
    # Table settings
    gb = GridOptionsBuilder.from_dataframe(final_rec)
    gb.configure_column('Ticker', editable = False)  
    gb.configure_column('Percentage', editable = False) # <- Non editable
    
    go = gb.build()
    
    # U know
    cols = st.columns(2)

    with cols[0]:
        d = st.slider("2. Укажите максимальную долю на один актив", min_value=0.1, max_value=1., step=0.1)
    with cols[1]:
        threshold = st.select_slider("3. Выберите ваш риск-профиль", ['Консервативный 🐣', 'Умеренный 🙊', 'Агрессивный 🦅'])
        
        threshold = {"Консервативный":0.001299, 
                     "Умеренный":0.002591, 
                     "Агрессивный":0.005104}[threshold[:-2]]
        
    st.markdown('')
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown('---')
    with cols[2]:
        st.markdown('---')
    
    with cols[1]:
        opt_but = st.empty()
        optimize = opt_but.button('Получить распределение с учётом параметров', on_click=change_text)
    result_info = st.empty()
     
    with st.spinner('Минуточку. Считаю-считаю...'):
        
        # Almost same as a previous
        if optimize:
            
            st.session_state.dff = inference_data
            
            n_securities = st.session_state.dff.shape[0]

            prices = st.session_state.dff['y_pred'].to_numpy()

            uncertainty = st.session_state.dff['total_var'].to_numpy()

            uncertainty = np.where(uncertainty>threshold, 1, uncertainty)
            
            allocs_creator = Minimizer(n_securities=n_securities, 
                                       divercity_ratio=d
                                      )

            allocs = allocs_creator.optimize(prices, uncertainty)

            st.session_state.dff.loc[:, 'Percentage'] = allocs
            st.session_state.dff.rename(columns={"full_name":"Ticker"}, inplace=True)


            st.markdown(f"<h2>{st.session_state.get_text}</h2>", unsafe_allow_html=True)
                
            final_rec = st.session_state.dff[allocs>0].reset_index(drop=True)[["Ticker", "Percentage"]]
            
            # Result table
            grid_return1 = AgGrid(final_rec,
                         reload_data = True,        
                         gridOptions=go, 
                         theme='streamlit', 
                         fit_columns_on_grid_load=True, 
                         height=300, 
                         custom_css={".ag-row-hover, .ag-active-item": {
                                        "color": "white !important"},
                                     ".ag-cell-inline-editing": {
                                        "color": "black !important"}
                                    })

            
            # Metric
            expected = sum(final_rec.Percentage.to_numpy()*\
                           st.session_state.dff.loc[allocs>0,
                            'y_pred'])*100
            cols = st.columns(2)
            
            
            with cols[0]:
                st.metric(label="Доходность предложенного портфеля",
                          value="".join([str(round(expected, 2)), "%"])
                         )
            
            # Download button    
            with cols[1]:
                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(final_rec)

                st.download_button(
                    label="Скачать портфель",
                    data=csv,
                    file_name='rec_portfolio.csv',
                    mime='text/csv',
                )

