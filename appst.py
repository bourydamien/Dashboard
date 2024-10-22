import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import requests

url = "https://9997-34-81-12-206.ngrok-free.app/predict_csv"
csv_file_path = "test.csv"
test = pd.read_csv(csv_file_path)

#-------------------------------------------------
if 'predictions' not in st.session_state:
    with st.spinner("Chargement des prédictions..."):
        # Envoi de la requête POST avec le fichier CSV
        with open(csv_file_path, 'rb') as file:
            files = {'file': (csv_file_path, file, 'text/csv')}
            response = requests.post(url, files=files)

        # Attendre la réponse et traiter les données reçues
        predictions = response.json()
        print(predictions)  # Debug: affiche la réponse de l'API
        st.session_state['predictions'] = predictions['predictions']['proba_class_1']
        test['proba'] = st.session_state['predictions']
else:
    # Si les prédictions sont déjà dans le state, les utiliser directement
    test['proba'] = st.session_state['predictions']

#--------------------------------------------------
st.title('Sélection de l\'ID du client')

# Stocker la sélection de l'utilisateur dans l'état
selected_id = st.selectbox('Veuillez sélectionner votre ID client', test['SK_ID_CURR'].unique())

# Bouton "OK" pour confirmer la sélection
if st.button('OK'):
    # Stocker les données du client sélectionné dans session_state
    st.session_state['selected_id'] = selected_id

# Une fois que l'utilisateur a cliqué sur "OK", recharger les informations
if 'selected_id' in st.session_state:
    client_data = test[test['SK_ID_CURR'] == st.session_state['selected_id']]
    print(client_data.head())  

    # Mettre à jour les variables
    revenu_particulier = client_data['AMT_INCOME_TOTAL'].values[0]  # Revenus totaux
    user_credit = client_data['AMT_CREDIT'].values[0]  # Montant du crédit
    cat_utilisateur = client_data['NAME_INCOME_TYPE'].values[0]  # Catégorie d'utilisateur

    # Vérifiez si 'proba' existe dans client_data
    if 'proba' in client_data.columns:
        value = client_data['proba'].values[0]  # Probabilité prédite
    else:
        value = None  # Ou une valeur par défaut

    # Afficher les informations du client sélectionné
    st.write(f"Vous avez sélectionné l'ID client : {selected_id}")
    st.write('Voici les informations associées à votre demande:')
    st.dataframe(client_data)
    if value < 0.5:
        st.write("Votre score est trop faible, le crédit risque de ne pas être accepté")
    else : 
        st.write("Votre score est bon, votre demande va être acceptée")
#-----------------------------------------------
def load_income_total():
    return pd.read_csv('income.csv')

# Fonction pour générer le graphique
def generate_graph(revenu_particulier , df_income):
    df = df_income

    # Créer les déciles
    deciles = pd.qcut(df['AMT_INCOME_TOTAL'], 10)
    intervals = deciles.cat.categories

    intervals_df = pd.DataFrame({
        'left': [interval.left for interval in intervals],
        'right': [interval.right for interval in intervals],
        'Intervalle': [f"{int(interval.left)} - {int(interval.right)}" for interval in intervals]
    })

    percentages = deciles.value_counts(normalize=True).sort_index() * 100
    intervals_df['Pourcentage'] = percentages.values

    # Trouver l'index correspondant à l'intervalle du revenu particulier
    highlight_idx = intervals_df[(intervals_df['left'] <= revenu_particulier) & (intervals_df['right'] >= revenu_particulier)]

    if highlight_idx.empty:
        # Si le revenu particulier n'est pas dans les intervalles, on trouve l'intervalle le plus proche
        highlight_idx = intervals_df[intervals_df['right'] >= revenu_particulier]

        if highlight_idx.empty:
            st.error("Le revenu particulier est en dehors des limites des intervalles de revenus.")
            return None
        
        highlight_idx = highlight_idx.index[-1]  # On prend le dernier index correspondant
    else:
        highlight_idx = highlight_idx.index[0]  # On prend le premier index correspondant

    # Créer le graphique
    fig = px.bar(intervals_df, x='Intervalle', y='Pourcentage',
                 title='Répartition des revenus de toutes les demandes',
                 labels={'Pourcentage': 'Pourcentage (%)'},
                 color='Pourcentage',
                 color_continuous_scale=px.colors.sequential.Blues)

    

    # Ajouter une annotation pour indiquer le revenu particulier
    fig.add_annotation(
        x=intervals_df.loc[highlight_idx, 'Intervalle'],
        y=10,  
        text=f'Vous êtes ici : {revenu_particulier} €<br>Intervalle : {intervals_df.loc[highlight_idx, "Intervalle"]}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

    return fig

def plot_income_by_type(df, user_income_type):
    income_by_type = df.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].mean().reset_index()

    income_by_type['NAME_INCOME_TYPE'] = income_by_type['NAME_INCOME_TYPE'].str.strip()
    user_income_type = user_income_type.strip()  

    # Définir les couleurs : rouge pour la catégorie de l'utilisateur, bleu pour les autres
    income_by_type['color'] = income_by_type['NAME_INCOME_TYPE'].apply(
        lambda x: 'red' if x == user_income_type else 'blue'
    )

    fig = px.bar(income_by_type, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL',
                 title='Revenu Moyen par Catégorie de Revenus',
                 labels={'AMT_INCOME_TOTAL': 'Revenu Moyen'},
                 color='color',  
                 color_discrete_sequence=['blue', 'red'])  # Définir les couleurs

    # Supprimer la légende
    fig.update_layout(showlegend=False)

    return fig

def plot_income_distribution_by_type(df, income_type, user_income):
    filtered_df = df[df['NAME_INCOME_TYPE'] == income_type]

    # Créer l'histogramme
    fig = px.histogram(filtered_df, x='AMT_INCOME_TOTAL', nbins=30,
                       title=f'Distribution des Revenus pour {income_type}',
                       labels={'AMT_INCOME_TOTAL': 'Revenu'},
                       color_discrete_sequence=px.colors.sequential.Blues_r,
                       opacity=0.9)

    # Ajouter une barre verticale pour le revenu de l'utilisateur
    fig.add_vline(x=user_income, line_width=3, line_dash="dash", line_color="red",
                  annotation_text="Votre Revenu", annotation_position="top right")

    # Personnalisation du layout
    fig.update_layout(
        xaxis_title='Revenu',
        yaxis_title='Nombre de Clients',
        title_x=0.5,  # Centrer le titre
        template='plotly_white',
        bargap=0.2  
    )

    return fig

def plot_credit_distribution(df, user_credit, gender_filter=None, family_status_filter=None):
    """
    Visualise la distribution des crédits (AMT_CREDIT) avec des filtres et la position de l'utilisateur.
    
    :param df: DataFrame contenant les colonnes 'AMT_CREDIT', 'CODE_GENDER', et 'NAME_FAMILY_STATUS'.
    :param user_credit: Montant du crédit de l'utilisateur.
    :param gender_filter: Filtre pour visualiser seulement un genre ('M', 'F', 'XNA' ou 'Tous').
    :param family_status_filter: Filtre pour visualiser seulement un état familial ('Tous', ou valeurs possibles de NAME_FAMILY_STATUS).
    :return: Graphique Plotly représentant la distribution des crédits et la position de l'utilisateur.
    """
    # Appliquer le filtre de genre si sélectionné
    if gender_filter and gender_filter != 'Tous':
        df = df[df['CODE_GENDER'] == gender_filter]
    
    # Appliquer le filtre d'état familial si sélectionné
    if family_status_filter and family_status_filter != 'Tous':
        df = df[df['NAME_FAMILY_STATUS'] == family_status_filter]
    
    # Créer un histogramme de la distribution des crédits
    fig = px.histogram(df, x='AMT_CREDIT', nbins=30, 
                       title=f'Distribution des Montants de Crédit ({gender_filter}, {family_status_filter})',
                       labels={'AMT_CREDIT': 'Montant du Crédit'},
                       color_discrete_sequence=['blue'], opacity=0.8)

    # Ajouter une ligne verticale pour indiquer le crédit de l'utilisateur
    fig.add_vline(x=user_credit, line_width=3, line_dash="dash", line_color="red",
                  annotation_text="Votre Crédit", annotation_position="top right")

    # Personnaliser le layout avec espacement entre les barres
    fig.update_layout(
        xaxis_title='Montant du Crédit',
        yaxis_title='Nombre de Demandes',
        title_x=0.5,  # Centrer le titre
        template='plotly_white',
        bargap=0.2  # Espacement entre les barres (20%)
    )

    return fig

def plot_credit_vs_income(df, user_credit, user_income, gender_filter=None, family_status_filter=None, target_filter=None):
    # Appliquer les filtres si spécifiés
    if gender_filter and gender_filter != 'Tous':
        df = df[df['CODE_GENDER'] == gender_filter]
    
    if family_status_filter and family_status_filter != 'Tous':
        df = df[df['NAME_FAMILY_STATUS'] == family_status_filter]
    
    if target_filter is not None:  # Filtrer par TARGET (1 pour accepté, 0 pour refusé)
        df = df[df['TARGET'] == target_filter]

    # Créer le scatter plot entre AMT_INCOME_TOTAL et AMT_CREDIT
    fig = px.scatter(df, x='AMT_INCOME_TOTAL', y='AMT_CREDIT', 
                     title="Relation entre Revenu et Crédit (Crédit Accepté vs Refusé)",
                     labels={'AMT_INCOME_TOTAL': 'Revenu Total', 'AMT_CREDIT': 'Montant du Crédit'},
                     color='NAME_INCOME_TYPE',  # Différencier les types de revenu par couleur
                     trendline='ols',  # Ajouter une ligne de tendance
                     color_discrete_sequence=px.colors.qualitative.Set1)  # Palette de couleurs

    # Ajouter un point pour l'utilisateur avec go.Scatter
    fig.add_trace(go.Scatter(
        x=[user_income], y=[user_credit],
        mode='markers+text',  # Permet d'afficher le point et le texte
        marker=dict(color='red', size=12),  # Point rouge pour l'utilisateur
        name="Votre Position",
        text=["Votre Position"],  # Ajouter du texte pour indiquer la position
        textposition="top center"  # Positionner le texte au-dessus du point
    ))

    return fig

#----------------------------------------------




df=load_income_total()
# Définir la couleur en fonction de la valeur
if value < 0.5:
    color = "red"
else:
    color = "blue"
st.markdown("<h1 style='text-align: center; color: black;'>Le score de crédit est :</h1>", unsafe_allow_html=True)
# Afficher la jauge sous forme de barre de progression
# Utilisation de `st.markdown` avec HTML pour personnaliser la jauge
st.markdown(f"""
    <div style="width: 100%; background-color: lightgray; border-radius: 10px;">
        <div style="width: {value*100}%; background-color: {color}; padding: 5px; border-radius: 10px;"></div>
    </div>
    <p style="text-align: center;">{value*100:.2f}%</p>
""", unsafe_allow_html=True)
st.title("Analyse des Revenus")

fig1 = generate_graph(revenu_particulier,df)

fig2=plot_income_by_type(df, cat_utilisateur)

fig3 = plot_income_distribution_by_type(df, cat_utilisateur, revenu_particulier)
if fig1 is not None:
    st.plotly_chart(fig1)
gender_filter = st.selectbox("Sélectionner le genre", options=["Tous", "M", "F", "XNA"])

# Sélectionner l'état familial
family_status_filter = st.selectbox("Sélectionner l'état familial", options=["Tous", "Single / not married", "Married", "Civil marriage", "Widow", "Separated", "Unknown"])

# Sélectionner si le crédit est accepté ou refusé
target_filter = st.selectbox("Crédit accepté ou refusé", options=[None, 1, 0], format_func=lambda x: 'Tous' if x is None else ('Crédit accepté' if x == 1 else 'Crédit refusé'))

fig4 = plot_credit_vs_income(df, user_credit, revenu_particulier, gender_filter, family_status_filter, target_filter)

# Afficher le graphique
st.plotly_chart(fig4)

if fig2 is not None:
    st.plotly_chart(fig2)

if fig3 is not None:
    st.plotly_chart(fig3)

st.title("Comment vous vous positionnez par rapport aux candidats actuels")

fig4 = plot_income_by_type(test, cat_utilisateur)

if fig4 is not None:
    st.plotly_chart(fig4)

# Interface utilisateur dans Streamlit
st.title("Visualisation Interactive de votre positionnement par rapport aux candidats réçents")

# Ajouter un filtre pour le genre basé sur 'CODE_GENDER'
gender_filter = st.selectbox("Filtrer par sexe :", options=['Tous', 'M', 'F', 'XNA'])

# Ajouter un filtre pour l'état familial basé sur 'NAME_FAMILY_STATUS'
family_status_filter = st.selectbox("Filtrer par état familial :", 
                                    options=['Tous', 'Single / not married', 'Married', 'Civil marriage', 
                                             'Widow', 'Separated', 'Unknown'])

# Ajouter une entrée pour le montant de crédit de l'utilisateur
user_credit = st.number_input("Montant du crédit de l'utilisateur", min_value=0, value=300000)

# Générer le graphique avec les filtres et le crédit de l'utilisateur
fig_credit_distribution = plot_credit_distribution(test, user_credit, gender_filter, family_status_filter)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig_credit_distribution)