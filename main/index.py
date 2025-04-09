import random
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from scipy.signal import find_peaks
from dateutil import parser
import numpy as np
import requests


def create_sleep_data(player_ids, dates):
    data = []
    for player_id in player_ids:
        for date in dates:
            data.append({
                'player_id': player_id,
                'date': date,
                'sleep_duration': np.random.normal(7, 1),
                'sleep_quality': np.random.randint(1, 6)
            })
    return pd.DataFrame(data)


@st.cache_data
def load_data():
    try:
        gps = pd.read_csv('E:/Competition-main (1)/Competition-main/Data/cfc_gps_data.csv', encoding='latin1')
        cps = pd.read_csv('E:/Competition-main (1)/Competition-main/Data/cfc_cps_data.csv', encoding='latin1')
        srestore = pd.read_csv('E:/Competition-main (1)/Competition-main/Data/cfc_store_data.csv', encoding='latin1')
        prority = pd.read_csv('E:/Competition-main (1)/Competition-main/Data/cfc_priority_data.csv', encoding='latin1')
        if 'player_id' not in gps.columns:
            num_rows = len(gps)
            player_ids_fictifs = np.random.choice(range(1, 101), size=num_rows)
            gps['player_id'] = player_ids_fictifs

        player_ids = gps['player_id'].unique()
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        sleep_df = create_sleep_data(player_ids, dates)
        # Filtrer les valeurs NaN dans opposition_full
        equipes_adverses = gps['opposition_full'].dropna().unique()
        equipes_mapping = {
            equipe: {
                'ligue': 'Premier League',
                'ville': np.random.choice(['Londres', 'Manchester', 'Liverpool', 'Birmingham', 'Newcastle'])
            }
            for equipe in equipes_adverses
        }
        player_bios_df = create_player_bios(player_ids, equipes_mapping)
        return gps, cps, srestore, prority, sleep_df, player_bios_df, equipes_mapping
    except FileNotFoundError:
        st.error("Le fichier cfc_gps_data.csv n'a pas été trouvé.")
        return None, None, None, None, None, None, None


def create_player_bios(player_ids, equipes_mapping):
    data = []
    nationalities = ['Française', 'Italienne', 'Espagnole', 'Anglaise', 'Allemande']
    postes = ['Attaquant', 'Milieu', 'Défenseur', 'Gardien']
    equipes = list(equipes_mapping.keys())
    for player_id in player_ids:
        equipe_joueur = np.random.choice(equipes)
        ligue_joueur = equipes_mapping[equipe_joueur]['ligue']
        data.append(
            {
                'player_id': player_id,
                'nationalité': random.choice(nationalities),
                'poste': random.choice(postes),
                'âge': random.randint(18, 35),
                'équipe': equipe_joueur,
                'Ligue': ligue_joueur
            }
        )
    return pd.DataFrame(data)


gps, cps, srestore, prority, sleep_df, player_bios_df, equipes_mapping = load_data()

if gps is None:
    st.stop()

prority['Target set'] = pd.to_datetime(prority['Target set'], format='%d/%m/%Y')
prority['Review Date'] = pd.to_datetime(prority['Review Date'], format='%d/%m/%Y')

st.markdown("<h1 style='text-align: center; color: blue;'>Analyse des Données de la Compétition</h1>",
            unsafe_allow_html=True)

# Sidebar pour les filtres
st.sidebar.header('Filtres')

# Filtre match/entrainement
match_entrainement = st.sidebar.radio('Type de session', ['Match', 'Entraînement'])

if match_entrainement == 'Match':
    filtered_gps = gps[gps['md_plus_code'].notna()].copy()
else:
    filtered_gps = gps[gps['md_minus_code'].notna()].copy()

# Filtres de date
start_date = st.sidebar.date_input('Date de début (GPS)', key='start_date_gps')
end_date = st.sidebar.date_input('Date de fin (GPS)', key='end_date_gps')

# Filtres d'équipe adverse
opponents = gps['opposition_full'].unique()
selected_opponents = st.sidebar.multiselect('Équipes adverses', opponents, default=opponents)

# Conversion des dates en objets datetime pandas
if not filtered_gps.empty and 'date' in filtered_gps.columns:
    filtered_gps['date'] = pd.to_datetime(filtered_gps['date'].apply(lambda x: parser.parse(x)), errors='coerce')

if gps is not None and 'date' in gps.columns:
    gps['date'] = pd.to_datetime(gps['date'].apply(parser.parse), errors='coerce')

# Filtrage des données
filtered_gps = filtered_gps[
    (filtered_gps['date'] >= pd.to_datetime(start_date)) &
    (filtered_gps['date'] <= pd.to_datetime(end_date)) &
    (filtered_gps['opposition_full'].isin(selected_opponents))
    ]

# Tabs pour organiser les sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['Performances', 'Capacity physique', 'Recovery status', 'Individual priority areas', 'Biographie',
     'Données brutes'])

# Tab 1: Performances

with tab1:
    st.markdown("<h2 style='color: green;'>Performances</h2>", unsafe_allow_html=True)
    # Distance parcourue (match vs entraînement)
    st.subheader('Distance parcourue (match vs entraînement)')
    gps['Session Type'] = 'Entraînement'
    gps.loc[gps['md_plus_code'].notna(), 'Session Type'] = 'Match'
    fig_box = px.box(gps, x='Session Type', y='distance', title='Distribution de la distance parcourue',
                     hover_data=['distance'])
    st.plotly_chart(fig_box, use_container_width=True, key='box_plot_distance_unique_1')  # Ajout de use_container_width
    mean_distance = gps.groupby('Session Type')['distance'].mean().reset_index()

    fig_bar = px.bar(mean_distance, x='Session Type', y='distance', title='Distance moyenne parcourue',
                     hover_data=['distance'])
    st.plotly_chart(fig_bar, use_container_width=True, key="distance_barchat_unique_1")  # Ajout de use_container_width
    # Zones de fréquence cardiaque (match vs entraînement)
    st.markdown("<h2 style='color: green;'>Zones de fréquence cardiaque (match vs entraînement)</h2>",
                unsafe_allow_html=True)
    hr_cols = ['hr_zone_1_hms', 'hr_zone_2_hms', 'hr_zone_3_hms', 'hr_zone_4_hms', 'hr_zone_5_hms']

    for col in hr_cols:
        gps[col] = pd.to_timedelta(gps[col]).dt.total_seconds()

    hr_means = gps.groupby('Session Type')[hr_cols].mean().reset_index()
    fig_group = go.Figure()

    for col in hr_cols:
        fig_group.add_trace(go.Bar(name=col, x=hr_means['Session Type'], y=hr_means[col],
                                   hovertemplate=f'Type de session: %{{x}}<br>{col}: %{{y}}'))

    fig_group.update_layout(title='Temps moyen passé dans chaque zone de fréquence cardiaque',
                            xaxis_title='Type de session', yaxis_title='Temps (secondes)')
    st.plotly_chart(fig_group, use_container_width=True,
                    key="hr_zones_grouped_unique_1")  # Ajout de use_container_width
    st.cache_data.clear()
    fig_stack = go.Figure()

    for col in hr_cols:
        fig_stack.add_trace(go.Bar(name=col, x=hr_means['Session Type'], y=hr_means[col],
                                   hovertemplate=f'Type de session: %{{x}}<br>{col}: %{{y}}'))
    fig_stack.update_layout(title='Répartition du temps total dans les zones de fréquence cardiaque',
                            xaxis_title='Type de session', yaxis_title='Temps (secondes)', barmode='stack')
    st.plotly_chart(fig_stack, use_container_width=True,
                    key="hr_zones_stacked_unique_1")  # Ajout de use_container_width

    # Accélérations/décélérations
    st.markdown("<h2 style='color: green;'>Accélérations/décélérations</h2>", unsafe_allow_html=True)
    accel_cols = ['accel_decel_over_2_5', 'accel_decel_over_3_5', 'accel_decel_over_4_5']

    gps[accel_cols] = gps[accel_cols].apply(pd.to_numeric, errors='coerce')
    accel_data = gps.groupby('date')[accel_cols].sum().reset_index()
    fig_accel = px.line(accel_data, x='date', y=accel_cols, title='Accélérations/décélérations au fil du temps',
                        hover_data=accel_cols)
    st.plotly_chart(fig_accel, use_container_width=True,
                    key="accel_decel_trend_unique_1")  # Ajout de use_container_width
    st.cache_data.clear()
    fig_accel_peaks = go.Figure()
    for col in accel_cols:
        fig_accel_peaks.add_trace(go.Scatter(x=accel_data['date'], y=accel_data[col], mode='lines', name=col,
                                             hovertemplate=f'Date: %{{x}}<br>{col}: %{{y}}'))
        peaks, _ = find_peaks(accel_data['accel_decel_over_3_5'].fillna(0), distance=7)
        fig_accel_peaks.add_trace(
            go.Scatter(x=accel_data['date'][peaks], y=accel_data['accel_decel_over_3_5'].fillna(0)[peaks],
                       mode='markers', marker=dict(color='red'), name='Pics',
                       hovertemplate=f'Date: %{{x}}<br>Pic: %{{y}}'))
        fig_accel_peaks.update_layout(title='Accélérations/décélérations au fil du temps (avec pics)',
                                      xaxis_title='Date',
                                      yaxis_title='Nombre d\'accélérations/décélérations')
        st.plotly_chart(fig_accel_peaks, use_container_width=True)  # Ajout de use_container_width

    for col in accel_cols:
        accel_data[f'{col}_7d_mean'] = accel_data[col].rolling(window=7, min_periods=1).mean()
        accel_data[f'{col}_14d_mean'] = accel_data[col].rolling(window=14, min_periods=1).mean()

    for col in accel_cols:
        plot_data = accel_data[['date', f'{col}_7d_mean', f'{col}_14d_mean']].dropna()

        fig_mean = px.line(
            plot_data,
            x='date',
            y=[f'{col}_7d_mean', f'{col}_14d_mean'],
            title=f'Moyennes mobiles des {col}',
            labels={'value': 'Nombre d\'accélérations/décélérations', 'date': 'Date'},
            hover_data={f'{col}_7d_mean': ':.2f', f'{col}_14d_mean': ':.2f'}
        )
        fig_mean.update_layout(
            legend_title_text='Moyenne mobile',
            hovermode='x unified'
        )
        st.plotly_chart(fig_mean, use_container_width=True, key=f"accel_decel_mean_{col}_unique_1")

    for col in accel_cols:
        accel_data[f'{col}_daily_diff'] = accel_data[col].diff()

    plot_data = accel_data[['date'] + [f'{col}_daily_diff' for col in accel_cols]].dropna()

    fig_diff = px.bar(
        plot_data,
        x='date',
        y=[f'{col}_daily_diff' for col in accel_cols],
        title='Variations quotidiennes des accélérations/décélérations',
        labels={'value': 'Variation quotidienne'},
        hover_data={f'{col}_daily_diff': ':.2f' for col in accel_cols}
    )
    fig_diff.update_layout(barmode='group')
    st.plotly_chart(fig_diff, use_container_width=True, key="accel_decel_diff_unique_1")

    # Distances à haute vitesse (match vs entraînement)
    st.markdown("<h2 style='color: green;'>Distances à haute vitesse (match vs entraînement)</h2>",
                unsafe_allow_html=True)
    high_speed_cols = ['distance_over_21', 'distance_over_24', 'distance_over_27']
    high_speed_data = gps.groupby('Session Type')[high_speed_cols].sum().reset_index()
    fig_group = go.Figure()

    for col in high_speed_cols:
        fig_group.add_trace(go.Bar(name=col, x=high_speed_data['Session Type'], y=high_speed_data[col],
                                   hovertemplate=f'Type de session: %{{x}}<br>{col}: %{{y}}'))
    fig_group.update_layout(title='Distances à haute vitesse (match vs entraînement)',
                            xaxis_title='Type de session', yaxis_title='Distance (m)')
    st.plotly_chart(fig_group, use_container_width=True,
                    key="high_speed_group_unique_1")  # Ajout de use_container_width
    fig_stack = go.Figure()

    for col in high_speed_cols:
        fig_stack.add_trace(go.Bar(name=col, x=high_speed_data['Session Type'], y=high_speed_data[col],
                                   hovertemplate=f'Type de session: %{{x}}<br>{col}: %{{y}}'))
    fig_stack.update_layout(title='Répartition des distances à haute vitesse', xaxis_title='Type de session',
                            yaxis_title='Distance (m)', barmode='stack')
    st.plotly_chart(fig_stack, use_container_width=True,
                    key="high_speed_stack_unique_1")  # Ajout de use_container_width

    # Zones de fréquence cardiaque (temps total)
    st.markdown("<h2 style='color: green;'>Temps passé dans les zones de fréquence cardiaque (total)</h2>",
                unsafe_allow_html=True)
    hr_data = gps.groupby('date')[hr_cols].sum().reset_index()
    fig_stack = go.Figure()

    for col in hr_cols:
        fig_stack.add_trace(
            go.Bar(name=col, x=hr_data['date'], y=hr_data[col], hovertemplate=f'Date: %{{x}}<br>{col}: %{{y}}'))
    fig_stack.update_layout(title='Répartition du temps dans les zones de fréquence cardiaque', xaxis_title='Date',
                            yaxis_title='Temps (secondes)', barmode='stack')
    st.plotly_chart(fig_stack, use_container_width=True, key="hr_zones_total_unique_1")  # Ajout de use_container_width
    fig_line = px.line(hr_data, x='date', y=hr_cols,
                       title='Temps dans les zones de fréquence cardiaque au fil du temps', hover_data=hr_cols)

    st.plotly_chart(fig_line, use_container_width=True,
                    key="hr_zones_timeline_unique_1")  # Ajout de use_container_width

    # Performances par équipe adverse
    st.markdown("<h2 style='color: green;'>Performances par équipe adverse</h2>", unsafe_allow_html=True)
    metrics = ['distance', 'peak_speed']

    for i, metric in enumerate(metrics):
        fig_box = px.box(gps, x='opposition_full', y=metric, title=f'Distribution de {metric} par équipe adverse',
                         hover_data=[metric])
        st.plotly_chart(fig_box, use_container_width=True, key=f"opp_box_{metric}_unique_1_{i}")
        mean_metric = gps.groupby('opposition_full')[metric].mean().reset_index()
        fig_bar = px.bar(mean_metric, x='opposition_full', y=metric, title=f'Moyenne de {metric} par équipe adverse',
                         hover_data=[metric])
        st.plotly_chart(fig_bar, use_container_width=True,
                        key=f"opp_bar_{metric}_unique_1_{i}")  # Ajout de use_container_width

    # Performances par saison
    st.markdown("<h2 style='color: green;'>Performances par saison</h2>", unsafe_allow_html=True)
    gps = gps.dropna(subset=['season'])
    gps['season'] = gps['season'].astype(str)
    metrics = ['distance', 'peak_speed', 'accel_decel_over_2_5', 'accel_decel_over_3_5', 'accel_decel_over_4_5']

    for i, metric in enumerate(metrics):
        mean_metric = gps.groupby('season')[metric].mean().reset_index()
        fig_line = px.line(mean_metric, x='season', y=metric, title=f'Évolution de {metric} par saison',
                           hover_data=[metric])
        st.plotly_chart(fig_line, use_container_width=True,
                        key=f"season_line_{metric}_unique_1_{i}")  # Ajout de use_container_width
        fig_bar = px.bar(mean_metric, x='season', y=metric, title=f'Moyenne de {metric} par saison',
                         hover_data=[metric])
        st.plotly_chart(fig_bar, use_container_width=True,
                        key=f"season_bar_{metric}_unique_1_{i}")  # Ajout de use_container_width
        fig_box = px.box(gps, x='season', y=metric, title=f'Distribution de {metric} par saison', hover_data=[metric])
        st.plotly_chart(fig_box, use_container_width=True,
                        key=f"season_box_{metric}_unique_1_{i}")  # Ajout de use_container_width

    # Relations entre les métriques
    st.markdown("<h2 style='color: green;'>Relations entre les métriques", unsafe_allow_html=True)
    fig_scatter = px.scatter(gps, x='peak_speed', y='distance_over_21',
                             title='Vitesse maximale vs Distance à haute vitesse',
                             hover_data=['peak_speed', 'distance_over_21'])
    st.plotly_chart(fig_scatter, use_container_width=True,
                    key="speed_vs_distance21_unique_1")  # Ajout de use_container_width
    fig_scatter3 = px.scatter(gps, x='peak_speed', y='distance_over_27',
                              title='Vitesse maximale vs Distance à haute vitesse',
                              hover_data=['peak_speed', 'distance_over_27'])
    st.plotly_chart(fig_scatter3, use_container_width=True,
                    key="speed_vs_distance27_unique_1")  # Ajout de use_container_width
    # KPIs
    st.markdown("<h2 style='color: green;'>Indicateurs de performance clés (KPI)", unsafe_allow_html=True)
    match_distances = gps[gps['md_plus_code'].notna()].groupby('md_plus_code')['distance'].sum()
    mean_match_distance = match_distances.mean()
    st.metric('Distance moyenne par match', f'{mean_match_distance:.2f} m')
    hr_high_cols = ['hr_zone_4_hms', 'hr_zone_5_hms']
    for col in hr_high_cols:
        gps[col] = pd.to_timedelta(gps[col]).dt.total_seconds()
        gps['hr_high_total'] = gps[hr_high_cols].sum(axis=1)
        mean_hr_high_total = gps['hr_high_total'].mean()
        st.metric('Temps total moyen en zone de fréquence cardiaque élevée', f'{mean_hr_high_total:.2f} s')

    # Distance totale parcourue au fil du temps
    st.markdown("<h2 style='color: green;'>Distance totale parcourue au fil du temps", unsafe_allow_html=True)
    gps['date'] = pd.to_datetime(gps['date'], format='%d/%m/%Y')
    gps['date'] = gps['date'].dt.date
    distance = gps.groupby('date')['distance'].sum()
    st.line_chart(distance)

    # Vitesse maximale par jour
    st.markdown("<h2 style='color: green;'>Vitesse maximale par jour", unsafe_allow_html=True)
    peak_speed = gps.groupby('date')['peak_speed'].max()
    st.bar_chart(peak_speed)
    # Charge du joueur par minutes
    st.markdown("<h2 style='color: green;'>Estimation de la charge du joueur par minutes", unsafe_allow_html=True)
    gps['day_duration'] = gps['day_duration'].fillna(0)
    gps['day_duration'] = gps['day_duration'].astype(float)
    gps['day_duration_minutes'] = gps['day_duration']

    # calculer une estimation de la charge par minutes
    gps['estimated_load_per_minute'] = (gps[high_speed_cols].sum(axis=1) + gps[accel_cols].sum(axis=1)) / gps[
        'day_duration_minutes']
    # Remplacer les valeurs infinies ou NaN par 0
    gps['estimated_load_per_minutes'] = gps['estimated_load_per_minute'].replace([float('inf'), -float('inf')],
                                                                                 0).fillna(0)

    # Creer les graphiques
    fig_estimated_load = px.line(gps, x='date', y='estimated_load_per_minute',
                                 title='Estimation de la charge du joueur par minute',
                                 hover_data=['estimated_load_per_minute'])
    st.plotly_chart(fig_estimated_load, use_container_width=True, key="player_load_unique_1")

    # Definition de seuil de charge
    seuil_bas = gps['estimated_load_per_minutes'].quantile(0.25)
    seuil_moyen = gps['estimated_load_per_minutes'].median()
    seuil_haut = gps['estimated_load_per_minutes'].quantile(0.75)


    # creer une colonne pour les alertes
    def assign_alert(load):
        if load > seuil_haut:
            return 'Haute charge'
        elif load > seuil_moyen:
            return 'Charge moyenne'
        elif load > seuil_bas:
            return 'Charge basse'
        else:
            return 'Charge normale'


    gps['alerte_level'] = gps['estimated_load_per_minutes'].apply(assign_alert)

    # creation du graphique avec des couleurs bassees sur les alertes
    fig_estimated_load = px.line(
        gps,
        x='date',
        y='estimated_load_per_minute',
        color='alerte_level',
        title='Estimation de la charge du joueur par minute avec alertes',
        color_discrete_map={
            'Haute charge': 'red',
            'Charge moyenne': 'orange',
            'Charge basse': 'yellow',
            'Charge normale': 'green'
        },
        hover_data=['estimated_load_per_minute', 'alerte_level']
    )
    st.plotly_chart(fig_estimated_load, use_container_width=True, key="player_load_alert_unique_1")
    # Afficher les statistiques de charge
    st.markdown("<h2 style='color: green;'>Statistiques de charge", unsafe_allow_html=True)
    st.write(f'Seuil bas: {seuil_bas:.2f}')
    st.write(f'Seuil moyen: {seuil_moyen:.2f}')
    st.write(f'Seuil haut: {seuil_haut:.2f}')
    st.write(f'Charge moyenne par minute: {gps["estimated_load_per_minute"].mean():.2f}')
    st.write(f'Charge maximale par minute: {gps["estimated_load_per_minute"].max():.2f}')
    st.write(f'Charge minimale par minute: {gps["estimated_load_per_minute"].min():.2f}')

    # Definir les seuils de risque ( a ajuster selon les besoins)
    seuil_risque_entrainement = gps['estimated_load_per_minute'].quantile(0.90)
    seuil_risque_blessure = gps['estimated_load_per_minute'].std()


    # creer une colonne pour les alertes de risque
    def assign_risk_alert(load):
        if load > seuil_risque_entrainement and load > seuil_risque_blessure:
            return 'Risque élevé de surentraînement et de blessure'
        elif load > seuil_risque_entrainement:
            return 'Risque de surentraînement'
        elif load > seuil_risque_blessure:
            return 'Risque de blessure'
        else:
            return 'Risque normal'


    gps['risk_alert_level'] = gps['estimated_load_per_minute'].apply(assign_risk_alert)
    fig_estimated_load = px.line(
        gps,
        x='date',
        y='estimated_load_per_minute',
        color='risk_alert_level',
        title='Estimation de la charge du joueur par minute avec alertes de risque',
        color_discrete_map={
            'Risque élevé de surentraînement et de blessure': 'red',
            'Risque de surentraînement': 'orange',
            'Risque de blessure': 'yellow',
            'Risque normal': 'green'
        },
        hover_data=['estimated_load_per_minute', 'risk_alert_level']
    )

    st.plotly_chart(fig_estimated_load, use_container_width=True, key="player_load_risk_unique_1")
    # Afficher les statistiques de risque
    st.markdown("<h2 style='color: green;'>Statistiques de risque", unsafe_allow_html=True)
    st.write(f'Seuil de risque de surentraînement: {seuil_risque_entrainement:.2f}')
    st.write(f'Seuil de risque de blessure: {seuil_risque_blessure:.2f}')
    st.write(f'Charge moyenne par minute: {gps["estimated_load_per_minute"].mean():.2f}')
    st.write(f'Charge maximale par minute: {gps["estimated_load_per_minute"].max():.2f}')
    st.write(f'Charge minimale par minute: {gps["estimated_load_per_minute"].min():.2f}')
with tab2:
    st.markdown("<h2 style='color: green;'>Capacités Physiques", unsafe_allow_html=True)
    # Filtres de mouvement, quality et expressions
    movements = cps['movement'].unique()
    qualities = cps['quality'].unique()
    expressions = cps['expression'].unique()
    selected_movements = st.sidebar.multiselect('Mouvements', movements, default=movements)
    selected_qualities = st.sidebar.multiselect('Qualités', qualities, default=qualities)
    selected_expression = st.sidebar.multiselect('Expressions', expressions, default=expressions)

    # Filtrage de données
    filtered_cps = cps[
        cps['movement'].isin(selected_movements) &
        cps['quality'].isin(selected_qualities) &
        cps['expression'].isin(selected_expression)
        ]

    # Affichage des données filtrées
    st.subheader('Données filtrées')
    st.dataframe(filtered_cps)
    # Graphique de tendance temporelle
    st.markdown("<h2 style='color: green;'>Tendances temporelles de BenchmarkPct", unsafe_allow_html=True)
    fig_trend = px.line(filtered_cps, x='ï»¿testDate', y='benchmarkPct', color='movement', facet_row='quality',
                        facet_col='expression', title='Tendances de BenchmarkPct', hover_data=['benchmarkPct'])
    st.plotly_chart(fig_trend, use_container_width=True,
                    key="benchmark_trend_main_unique_1")  # Ajout de use_container_width
    # Graphique de comparaison
    st.markdown("<h2 style='color: green;'>Comparaison de BenchmarkPct", unsafe_allow_html=True)
    fig_compare = px.box(filtered_cps, x='movement', y='benchmarkPct', color='quality', facet_col='expression',
                         title='Comparaison de BenchmarkPct', hover_data=['benchmarkPct'])
    st.plotly_chart(fig_compare, use_container_width=True,
                    key="benchmark_compare_main_unique_1")  # Ajout de use_container_width

    # KPIs
    st.markdown("<h2 style='color: green;'>Indicateurs de performance clés (KPI)", unsafe_allow_html=True)
    mean_benchmarks = filtered_cps.groupby(['movement', 'quality', 'expression'])['benchmarkPct'].mean().reset_index()
    st.dataframe(mean_benchmarks)
with tab3:
    st.markdown("<h2 style='color: green;'>Analyse du statut de récupération", unsafe_allow_html=True)
    # Filtres de date
    start_Date = st.sidebar.date_input('Date de début')
    end_Date = st.sidebar.date_input('Date de fin')
    # Filtres de categorie et metrique
    categories = srestore['category'].unique()
    metrics = srestore['metric'].unique()
    selected_categories = st.sidebar.multiselect('Catégories', categories, default=categories)
    selected_metrics = st.sidebar.multiselect('Métriques', metrics, default=metrics)
    filtered_srestore = srestore[
        srestore['category'].isin(selected_categories) &
        srestore['metric'].isin(selected_metrics)
        ]

    # Filtrage de donnees
    st.subheader('Données filtrées')
    st.dataframe(filtered_srestore)

    # Graphique de tendance temporelle
    st.markdown("<h2 style='color: green;'>Tendances temporelles des valeurs de récupération", unsafe_allow_html=True)
    fig_trend = px.line(filtered_srestore, x='sessionDate', y='value', color='metric', facet_col='category',
                        title='Tendances des valeurs de récupération', hover_data=['value'])
    st.plotly_chart(fig_trend, use_container_width=True,
                    key="recovery_trend_main_unique_1")  # Ajout de use_container_width

    # Graphique de comparaison
    st.markdown("<h2 style='color: green;'>Comparaison des valeurs de récupération", unsafe_allow_html=True)
    fig_compare = px.box(filtered_srestore, x='category', y='value', color='metric',
                         title='Comparaison des valeurs de récupération', hover_data=['value'])
    st.plotly_chart(fig_compare, use_container_width=True)  # Ajout de use_container_width
    st.markdown("<h2 style='color: green;'>Données de sommeil", unsafe_allow_html=True)
    st.dataframe(sleep_df)

    # Graphiques de sommeil
    st.markdown("<h2 style='color: green;'>Qualité du sommeil au fil du temps", unsafe_allow_html=True)
    fig_sleep_quality = px.line(sleep_df, x='date', y='sleep_quality', color='player_id')
    st.plotly_chart(fig_sleep_quality, use_container_width=True, key="sleep_quality_unique_1")

    # Analyse de la complétude
    if '_completeness' in filtered_srestore['metric'].values:
        st.subheader('Analyse de la complétude')
        completeness_data = filtered_srestore[filtered_srestore['metric'].str.endswith('_completeness')]
        fig_completeness = px.line(completeness_data, x='sessionDate', y='value', color='category',
                                   title='Complétude des données de récupération', hover_data=['value'])
        st.plotly_chart(fig_completeness, use_container_width=True,
                        key="completeness_chart_unique_1")  # Ajout de use_container_width

    # Analyse du score composite
    if '_composite' in filtered_srestore['metric'].values:
        st.subheader('Analyse du score composite')
        composite_data = filtered_srestore[filtered_srestore['metric'].str.endswith('_composite')]
        fig_composite = px.line(composite_data, x='sessionDate', y='value', color='category',
                                title='Score composite de récupération', hover_data=['value'])
        st.plotly_chart(fig_composite, use_container_width=True,
                        key="composite_chart_unique_1")  # Ajout de use_container_width

    # Analyse du score total
    if 'emboss_baseline_score' in filtered_srestore['metric'].values:
        st.markdown("<h2 style='color: green;'>Analyse du score total de récupération", unsafe_allow_html=True)
        total_score_data = filtered_srestore[filtered_srestore['metric'] == 'emboss_baseline_score']
        fig_total_score = px.line(total_score_data, x='sessionDate', y='value', title='Score total de récupération',
                                  hover_data=['value'])
        st.plotly_chart(fig_total_score, use_container_width=True,
                        key="total_score_chart_unique_1")  # Ajout de use_container_width

with tab4:
    st.markdown("<h2 style='color: green;'>Individual priority areas", unsafe_allow_html=True)
    # Filtres de categorie, zone, type de performance et suivi
    Categories = prority['Category'].unique()
    areas = prority['Area'].unique()
    performances_types = prority['Performance Type'].unique()
    tracking_statuses = prority['Tracking'].unique()

    selected_Categories = st.sidebar.multiselect('Catégories', Categories, default=Categories)
    selected_areas = st.sidebar.multiselect('Zones', areas, default=areas)
    selected_performaces_types = st.sidebar.multiselect('Types de performances', performances_types,
                                                        default=performances_types)
    selected_tracking_statuses = st.sidebar.multiselect('Etat de suivi', tracking_statuses, default=tracking_statuses)

    filtered_prority = prority[
        prority['Category'].isin(selected_Categories) &  # Correction ici
        prority['Area'].isin(selected_areas) &  # Correction ici
        prority['Performance Type'].isin(selected_performaces_types) &  # Correction ici
        prority['Tracking'].isin(selected_tracking_statuses)  # Correction ici
        ]

    # Affichage des données filtrées
    st.subheader('Priorités filtrées')
    st.dataframe(filtered_prority)

    # Répartition des catégories
    st.markdown("<h2 style='color: green;'>Répartition des catégories", unsafe_allow_html=True)
    category_counts = filtered_prority['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'count']  # Renommer les colonnes pour plus de clarté.

    fig_categories = px.bar(category_counts, x='Category', y='count', title='Répartition des priorités par catégorie',
                            labels={'Category': 'Catégorie', 'count': 'Nombre de priorités'}, hover_data=['count'])
    st.plotly_chart(fig_categories, use_container_width=True,
                    key="priority_categories_main_unique_1")  # Ajout de use_container_width
    # Répartition des zones
    st.markdown("<h2 style='color: green;'>Répartition des zones", unsafe_allow_html=True)
    area_counts = filtered_prority['Area'].value_counts().reset_index()
    area_counts_columns = ['Area', 'count']

    fig_areas = px.bar(area_counts, x='Area', y='count', title='Répartition des priorités par zone',
                       labels={'Area': 'Zone', 'count': 'Nombre de priorités'}, hover_data=['count'])
    st.plotly_chart(fig_areas, use_container_width=True,
                    key="priority_areas_main_unique_1")  # Ajout de use_container_width

    # Répartition des types de performance
    st.markdown("<h2 style='color: green;'>Répartition de type des performances", unsafe_allow_html=True)
    performance_type_counts = filtered_prority['Performance Type'].value_counts().reset_index()
    performance_type_counts.columns = ['Performance Type', 'count']  # Renommer les colonnes pour plus de clarté.
    fig_performance_types = px.bar(performance_type_counts, x='Performance Type', y='count',
                                   title='Répartition des priorités par type de performance',
                                   labels={'Performance Type': 'Type de performance', 'count': 'Nombre de priorités'},
                                   hover_data=['count'])
    st.plotly_chart(fig_performance_types, use_container_width=True,
                    key="performance_types_main_unique_1")  # Ajout de use_container_width

    # Suivi des priorités
    st.markdown("<h2 style='color: green;'>Suivi des priorités", unsafe_allow_html=True)
    tracking_counts = filtered_prority['Tracking'].value_counts().reset_index()
    tracking_counts_columns = ['Tracking', 'count']

    fig_tracking = px.bar(tracking_counts, x='Tracking', y='count', title='Suivi des priorités',
                          labels={'Tracking': 'État de suivi', 'count': 'Nombre de priorités'}, hover_data=['count'])
    st.plotly_chart(fig_tracking, use_container_width=True,
                    key="tracking_status_main_unique_1")  # Ajout de use_container_width

    # Dates de révision
    st.markdown("<h2 style='color: green;'>Dates de révision", unsafe_allow_html=True)
    fig_review_dates = px.histogram(filtered_prority, x='Review Date', title='Dates de révision des priorités',
                                    hover_data=['Review Date'])
    st.plotly_chart(fig_review_dates, use_container_width=True,
                    key="review_dates_main_unique_1")  # Ajout de use_container_width

with tab5:
    st.markdown("<h2 style='color: green;'>Biographie du joueur", unsafe_allow_html=True)
    selected_player_id = st.selectbox("Sélectionner un joueur", player_bios_df['player_id'].unique())
    selected_player_bio = player_bios_df[player_bios_df['player_id'] == selected_player_id].iloc[0]
    st.write(f"**ID du joueur:** {selected_player_bio['player_id']}")
    st.write(f"**Nationalité:** {selected_player_bio['nationalité']}")
    st.write(f"**Poste:** {selected_player_bio['poste']}")
    st.write(f"**Âge:** {selected_player_bio['âge']}")
    st.write(f"**Équipe:** {selected_player_bio['équipe']}")

with tab6:
    st.markdown("<h2 style='color: green;'>Données brutes", unsafe_allow_html=True)
    st.dataframe(filtered_gps)
