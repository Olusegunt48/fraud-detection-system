import streamlit as st
import requests
import pandas as pd
import json
import io
import os


API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')


def get_model_info():
    try:
        response = requests.get(f'{API_URL}/model-info')
        return response.json()
    except Exception:
        return None


def predict_transaction(transaction_data):
    try:
        response = requests.post(
            f'{API_URL}/predict',
            json=transaction_data
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}


FEATURE_ORDER = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']


st.set_page_config(
    page_title='Fraud Detection System',
    page_icon='🔍',
    layout='wide'
)

st.title('🔍 Credit Card Fraud Detection System')
st.markdown('Analyse individual transactions or upload a CSV file for batch predictions.')

model_info = get_model_info()

if model_info:
    st.sidebar.success('✅ API Connected')
    st.sidebar.markdown(f"**Model:** {model_info['model']}")
    st.sidebar.markdown(f"**Features:** {model_info['total_features']}")
else:
    st.sidebar.error('❌ API Offline — Start FastAPI first')
    st.stop()

st.sidebar.markdown('---')

mode = st.sidebar.radio(
    'Select Mode',
    ['Single Transaction', 'Batch Upload']
)

st.markdown('---')

if mode == 'Single Transaction':
    st.subheader('Single Transaction Analysis')

    st.sidebar.header('Transaction Details')

    amount = st.sidebar.number_input('Amount ($)', min_value=0.0, value=100.0, step=0.01)
    time   = st.sidebar.number_input('Time (seconds)', min_value=0.0, value=0.0, step=1.0)

    st.sidebar.markdown('---')
    st.sidebar.subheader('V Features')

    v_features = {}
    col1, col2 = st.sidebar.columns(2)

    for i in range(1, 29):
        if i % 2 == 0:
            v_features[f'V{i}'] = col2.number_input(
                f'V{i}', value=0.0, format='%.4f', key=f'V{i}'
            )
        else:
            v_features[f'V{i}'] = col1.number_input(
                f'V{i}', value=0.0, format='%.4f', key=f'V{i}'
            )

    transaction_data = {**v_features, 'Amount': amount, 'Time': time}

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader('Transaction Summary')
        st.markdown(f"**Amount:** ${amount:,.2f}")
        st.markdown(f"**Time:** {time:,.0f} seconds")
        non_zero = {k: v for k, v in v_features.items() if v != 0.0}
        if non_zero:
            st.markdown('**Non-zero V Features:**')
            for k, v in non_zero.items():
                st.markdown(f"  - {k}: {v:.4f}")
        else:
            st.info('All V features are set to 0.0')

    with col_right:
        st.subheader('Prediction')

        if st.button('🔍 Analyse Transaction', use_container_width=True):
            with st.spinner('Analysing transaction...'):
                result = predict_transaction(transaction_data)

            if 'error' in result:
                st.error(f"Error: {result['error']}")

            elif result['label'] == 'Fraud':
                st.error('🚨 FRAUDULENT TRANSACTION DETECTED')
                st.metric(
                    label='Fraud Probability',
                    value=f"{result['fraud_probability'] * 100:.2f}%"
                )

            else:
                st.success('✅ LEGITIMATE TRANSACTION')
                st.metric(
                    label='Fraud Probability',
                    value=f"{result['fraud_probability'] * 100:.2f}%"
                )

            st.markdown('---')
            st.subheader('Raw API Response')
            st.json(result)


elif mode == 'Batch Upload':
    st.subheader('Batch Transaction Analysis')

    st.info(
        'Upload a CSV file containing transaction data. '
        'The file must have these columns: '
        'V1 through V28, Amount, and Time.'
    )

    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown(f"**Rows uploaded:** {len(df):,}")
        st.markdown('**Preview (first 5 rows):**')
        st.dataframe(df.head())

        missing_cols = [col for col in FEATURE_ORDER if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        if st.button('🔍 Analyse All Transactions', use_container_width=True):
            results = []
            progress = st.progress(0)
            status   = st.empty()

            for i, row in df.iterrows():
                transaction_data = {feature: row[feature] for feature in FEATURE_ORDER}
                result = predict_transaction(transaction_data)
                results.append(result)

                progress.progress((i + 1) / len(df))
                status.text(f'Analysing transaction {i + 1} of {len(df)}...')

            progress.empty()
            status.empty()

            results_df = pd.DataFrame(results)
            df['Prediction']       = results_df['prediction'].values
            df['Label']            = results_df['label'].values
            df['Fraud_Probability'] = results_df['fraud_probability'].values

            st.markdown('---')
            st.subheader('Results Summary')

            total     = len(df)
            fraud     = len(df[df['Label'] == 'Fraud'])
            legit     = len(df[df['Label'] == 'Legitimate'])

            col1, col2, col3 = st.columns(3)
            col1.metric('Total Transactions', f'{total:,}')
            col2.metric('Fraudulent',         f'{fraud:,}')
            col3.metric('Legitimate',         f'{legit:,}')

            st.markdown('---')
            st.subheader('Detailed Results')

            def colour_rows(row):
                if row['Label'] == 'Fraud':
                    return ['background-color: #c0392b; color: #ffffff'] * len(row)
                else:
                    return ['background-color: #1e8449; color: #ffffff'] * len(row)

            st.dataframe(
                df[['Amount', 'Time', 'Label', 'Fraud_Probability']].style.apply(
                    colour_rows, axis=1
                )
            )   

            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download Results as CSV',
                data=csv_output,
                file_name='fraud_detection_results.csv',
                mime='text/csv',
                use_container_width=True
            )

st.markdown('---')
st.caption('Fraud Detection System — Powered by XGBoost & FastAPI')