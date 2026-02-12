import pandas as pd
import numpy as np
import math
import random
import os
import requests

from scipy.stats import poisson
from collections import Counter
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, log_loss

from sklearn.pipeline import make_pipeline

import copy

# --- CONFIGURATION ---
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = 'http://api.football-data.org/v4/competitions/PL/matches'
HEADERS = {'X-Auth-Token': API_KEY}
HISTORY_FILE = 'updated_results.csv'  #training data
OUTPUT_FILE = 'predicted_table.csv'   #output data


TEAM_MAPPING = {
    "Wolverhampton Wanderers FC": "Wolves",
    "Manchester United FC": "Man United",
    "Chelsea FC": "Chelsea",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Fulham FC": "Fulham",
    "Arsenal FC": "Arsenal",
    "Sunderland AFC": "Sunderland",
    "Newcastle United FC": "Newcastle",
    "Crystal Palace FC": "Crystal Palace",
    "Manchester City FC": "Man City",
    "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Leeds United FC": "Leeds",
    "AFC Bournemouth": "Bournemouth"
}

def fetch_api_data(status_list):
    """
    Fetches matches where status is IN the provided list.
    Example: ['FINISHED'] or ['SCHEDULED', 'TIMED', 'POSTPONED']
    """
    # Request the specific season (2025) to ensure we get all 380 games
    response = requests.get(BASE_URL, headers=HEADERS, params={'season': 2025}) 
    data = response.json()
    
    matches = []
    for match in data.get('matches', []):
        # FIX: Check if status is in the list (e.g., allow TIMED and POSTPONED)
        if match['status'] in status_list:
            
            # Logic for FUTURE games (Scheduled, Timed, Postponed)
            if match['status'] != 'FINISHED':
                 matches.append({
                    'Date': match['utcDate'][:10],
                    'HomeTeam': TEAM_MAPPING.get(match['homeTeam']['name'], match['homeTeam']['name']),
                    'AwayTeam': TEAM_MAPPING.get(match['awayTeam']['name'], match['awayTeam']['name']),
                    'FTHG': 0, 
                    'FTAG': 0, 
                    'FTR': None 
                })
            
            # Logic for PAST games (Finished)
            else:
                matches.append({
                    'Date': match['utcDate'][:10],
                    'HomeTeam': TEAM_MAPPING.get(match['homeTeam']['name'], match['homeTeam']['name']),
                    'AwayTeam': TEAM_MAPPING.get(match['awayTeam']['name'], match['awayTeam']['name']),
                    'FTHG': match['score']['fullTime']['home'],
                    'FTAG': match['score']['fullTime']['away'],
                    'FTR': 'H' if match['score']['winner'] == 'HOME_TEAM' else 
                           ('A' if match['score']['winner'] == 'AWAY_TEAM' else 'D')
                })
    return pd.DataFrame(matches)

# --- STEP 1: LOAD & UPDATE HISTORY ---
print("1. Loading historical data...")
try:
    history_df = pd.read_csv(HISTORY_FILE)
    last_date = pd.to_datetime(history_df['Date']).max()
except FileNotFoundError:
    print("   No history file found. Starting fresh.")
    history_df = pd.DataFrame() 
    last_date = datetime(2020, 1, 1) 


print("2. Fetching recent results from API...")
# Fetch ALL finished matches for current season
current_season_df = fetch_api_data('FINISHED')

# Filter for games that are NEW (Date > last_date in CSV)
if not history_df.empty:
    current_season_df['Date'] = pd.to_datetime(current_season_df['Date'])
    new_games = current_season_df[current_season_df['Date'] > last_date]
else:
    new_games = current_season_df

if not new_games.empty:
    print(f"   Found {len(new_games)} new games. Updating history...")
    # Standardize format before appending
    new_games['Date'] = new_games['Date'].dt.strftime('%m/%d/%Y')
    history_df = pd.concat([history_df, new_games], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)
else:
    print("   No new games to add.")


print("3. Retraining Model...")
# Ensure Date is in datetime format
history_df['Date'] = pd.to_datetime(history_df['Date'], dayfirst=False)
history_df = history_df.sort_values('Date')

# Create a 'Season' column (eg. 2022, 2023)
# If Month is August (8) or later, it's the start of a new season year
# If Month is Jan-May, it belongs to the year prior (the start of that season)
history_df['Season'] = history_df['Date'].apply(lambda x: x.year if x.month > 7 else x.year - 1)



# WEIGHTED ELO
def calculate_elo_ratings_weighted(df, k_factor=20, start_rating=1500):
    ratings = {}
    home_elo, away_elo = [], []
    
    for _, row in df.iterrows():
        h_team, a_team, res = row['HomeTeam'], row['AwayTeam'], row['FTR']
        goal_diff = abs(row['FTHG'] - row['FTAG'])
        
        h_rate = ratings.get(h_team, start_rating)
        a_rate = ratings.get(a_team, start_rating)
        
        home_elo.append(h_rate)
        away_elo.append(a_rate)
        
        exp_home = 1 / (1 + 10**((a_rate - h_rate) / 400))
        exp_away = 1 - exp_home
        
        if res == 'H': act_h, act_a = 1, 0
        elif res == 'D': act_h, act_a = 0.5, 0.5
        else: act_h, act_a = 0, 1
            
        if res == 'D': mov_mult = 1
        else:
            mov_mult = math.log(goal_diff + 1) * 2.2 / ((h_rate - a_rate) * 0.001 + 2.2)
            try: mov_mult = max(1, mov_mult)
            except: mov_mult = 1

        shift = k_factor * mov_mult * (act_h - exp_home)
        ratings[h_team] = h_rate + shift
        ratings[a_team] = a_rate - shift
        
    df['Home_Elo'] = home_elo
    df['Away_Elo'] = away_elo
    return df
history_df = calculate_elo_ratings_weighted(history_df)

# REST DAYS
def calculate_rest_days(df):
    home_dates = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'})
    away_dates = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
    all_dates = pd.concat([home_dates, away_dates]).sort_values('Date')
    
    all_dates['Last_Match'] = all_dates.groupby('Team')['Date'].shift(1)
    all_dates['Rest_Days'] = (all_dates['Date'] - all_dates['Last_Match']).dt.days
    all_dates['Rest_Days'] = all_dates['Rest_Days'].fillna(7).clip(upper=14)
    
    df = pd.merge(df, all_dates.rename(columns={'Team': 'HomeTeam', 'Rest_Days': 'Home_Rest_Days'})[['Date', 'HomeTeam', 'Home_Rest_Days']], on=['Date', 'HomeTeam'], how='left')
    df = pd.merge(df, all_dates.rename(columns={'Team': 'AwayTeam', 'Rest_Days': 'Away_Rest_Days'})[['Date', 'AwayTeam', 'Away_Rest_Days']], on=['Date', 'AwayTeam'], how='left')
    return df
history_df = calculate_rest_days(history_df)

# SEASON STATS
def calculate_season_stats(df):
    h_df = df[['Date', 'Season', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']].rename(columns={'HomeTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA'})
    h_df['Pts'] = h_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_df = df[['Date', 'Season', 'AwayTeam', 'FTAG', 'FTHG', 'FTR']].rename(columns={'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA'})
    a_df['Pts'] = a_df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    stats = pd.concat([h_df, a_df]).sort_values('Date')
    
    stats['Season_PPG'] = stats.groupby(['Season', 'Team'])['Pts'].transform(lambda x: x.expanding().mean().shift(1))
    stats['Avg_GF'] = stats.groupby(['Season', 'Team'])['GF'].transform(lambda x: x.expanding().mean().shift(1))
    stats['Avg_GA'] = stats.groupby(['Season', 'Team'])['GA'].transform(lambda x: x.expanding().mean().shift(1))
    return stats

stats_long = calculate_season_stats(history_df)
stats_sub = stats_long[['Date', 'Team', 'Avg_GF', 'Avg_GA', 'Season_PPG']].copy()

history_df = pd.merge(history_df, stats_sub.rename(columns={'Team': 'HomeTeam', 'Avg_GF': 'Home_Avg_GF', 'Avg_GA': 'Home_Avg_GA', 'Season_PPG': 'Home_Season_PPG'}), on=['Date', 'HomeTeam'], how='left')
history_df = pd.merge(history_df, stats_sub.rename(columns={'Team': 'AwayTeam', 'Avg_GF': 'Away_Avg_GF', 'Avg_GA': 'Away_Avg_GA', 'Season_PPG': 'Away_Season_PPG'}), on=['Date', 'AwayTeam'], how='left')

# EMA FORM
def calculate_ema_form(df, span=5):
    h_df = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    h_df['Pts'] = h_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_df = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    a_df['Pts'] = a_df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    stats = pd.concat([h_df, a_df]).sort_values('Date')
    stats['Form_EMA'] = stats.groupby('Team')['Pts'].transform(lambda x: x.ewm(span=span, min_periods=1).mean().shift(1))
    return stats[['Date', 'Team', 'Form_EMA']]

form_stats = calculate_ema_form(history_df)
history_df = pd.merge(history_df, form_stats.rename(columns={'Team': 'HomeTeam', 'Form_EMA': 'Home_Form_EMA'}), on=['Date', 'HomeTeam'], how='left')
history_df = pd.merge(history_df, form_stats.rename(columns={'Team': 'AwayTeam', 'Form_EMA': 'Away_Form_EMA'}), on=['Date', 'AwayTeam'], how='left')

# POISSON PROBABILITIES
cols_to_fill = {
    'Home_Season_PPG': 1.37, 'Away_Season_PPG': 1.37,
    'Home_Form_EMA': 1.37, 'Away_Form_EMA': 1.37,
    'Home_Avg_GF': 1.4, 'Away_Avg_GF': 1.4,
    'Home_Avg_GA': 1.4, 'Away_Avg_GA': 1.4
}
history_df = history_df.fillna(cols_to_fill)

def calculate_poisson_features(row):
    lambda_home = (row['Home_Avg_GF'] + row['Away_Avg_GA']) / 2
    lambda_away = (row['Away_Avg_GF'] + row['Home_Avg_GA']) / 2
    
    prob_home_win = 0
    prob_draw = 0
    prob_away_win = 0
    
    for h in range(6): 
        for a in range(6): 
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            if h > a:
                prob_home_win += p
            elif h == a:
                prob_draw += p
            else:
                prob_away_win += p
                
    return pd.Series([prob_home_win, prob_draw, prob_away_win])

history_df[['Poisson_H', 'Poisson_D', 'Poisson_A']] = history_df.apply(calculate_poisson_features, axis=1)

# Other FEATURES
history_df['Elo_Diff'] = history_df['Home_Elo'] - history_df['Away_Elo']
history_df['Rest_Diff'] = history_df['Home_Rest_Days'] - history_df['Away_Rest_Days']
history_df['Form_Diff'] = history_df['Home_Form_EMA'] - history_df['Away_Form_EMA']
history_df['Att_vs_Def'] = history_df['Home_Avg_GF'] - history_df['Away_Avg_GA']

features = [
    'Home_Season_PPG', 'Away_Season_PPG',
    'Home_Elo', 'Away_Elo', 'Elo_Diff',
    'Home_Form_EMA', 'Away_Form_EMA', 'Form_Diff',
    'Rest_Diff',
    'Home_Avg_GF', 'Away_Avg_GF', 'Att_vs_Def',
    'Poisson_H', 'Poisson_D', 'Poisson_A' 
]
target = 'FTR'

X_train = history_df[features]
le = LabelEncoder()
y_train_encoded = le.fit_transform(history_df[target])

# HYPERPARAMETER TUNING
print("Grid Search to optimize model parameters")

# the Grid
param_grid = {
    'max_depth': [3, 4, 5],             # Shallower trees often generalize better
    'learning_rate': [0.01, 0.03, 0.05], # Test different learning speeds
    'n_estimators': [300, 500, 700],    # Test different numbers of trees
    'min_child_weight': [1, 3, 5],      # Control overfitting
    'subsample': [0.8],                 
    'colsample_bytree': [0.8]           
}

# Setup the Base Model
xgb_base = xgb.XGBClassifier(
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1
)

# Setup Time-Series Validation (To Prevent future leakage)
tscv = TimeSeriesSplit(n_splits=3)

# Run the Search
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

# Fit on TRAINING data
grid_search.fit(X_train, y_train_encoded)

# Extract Best Model
best_model = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)



print("Training Logistic Regression Model...")

# A. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# B. Initialize and Fit the Model
lr_model = LogisticRegression(
    C=0.1, 
    solver='lbfgs', 
    max_iter=1000, 
    multi_class='multinomial',
    random_state=42
)

lr_model.fit(X_scaled, y_train_encoded)



def build_current_standings(df, current_date_str=None):
    if current_date_str is None:
        current_date = pd.Timestamp.now().normalize()
    else:
        current_date = pd.to_datetime(current_date_str)
        
    print(f"Building standings up to {current_date.strftime('%Y-%m-%d')}...")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    # Get only 2025 matches that have already happened
    season_df = df[
        (df['Season'] == 2025) & 
        (df['Date'] < current_date)
    ].sort_values('Date').reset_index(drop=True)
    
    # Initialize the dictionary for all teams found in this season
    all_teams = pd.concat([season_df['HomeTeam'], season_df['AwayTeam']]).unique()
    
    # Structure: { 'Team': { 'Points': 0, 'GD': 0, ... } }
    stats = {team: {
        'Points': 0, 
        'GD': 0, 
        'Elo': 1500,        # Default, will update if history exists
        'Form': 1.0, 
        'GF_Total': 0,      # Temp for calculation
        'GA_Total': 0,      # Temp for calculation
        'Matches': 0,       # Temp for calculation
        'Last_Match_Date': pd.Timestamp('2025-08-01') # Default start of season
    } for team in all_teams}
    
    # ---------------------------------------------------------
    # 2. POPULATE ELO FROM HISTORY (OPTIONAL BUT RECOMMENDED)
    # ---------------------------------------------------------
    # Ideally, we grab the Elo from the very last match each team played in the DATAFRAME
    # because the dataframe already has the calculated "Home_Elo" and "Away_Elo".
    # This saves us from re-calculating Elo from 1993.
    
    for team in all_teams:
        # Find the last game this team played (Home or Away)
        last_game = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
            (df['Date'] < current_date)
        ].sort_values('Date').tail(1)
        
        if not last_game.empty:
            row = last_game.iloc[0]
            if row['HomeTeam'] == team:
                # We want the POST-match Elo. 
                # The dataframe usually stores PRE-match Elo.
                # So we take the Pre-match Elo + the update from that game.
                # Simplified: We will just trust the Pre-Match Elo of the *Next* game 
                # or just use the Pre-Match Elo of the current game for safety.
                stats[team]['Elo'] = row['Home_Elo'] 
                stats[team]['Form'] = row['Home_Form_EMA']
            else:
                stats[team]['Elo'] = row['Away_Elo']
                stats[team]['Form'] = row['Away_Form_EMA']

    # ---------------------------------------------------------
    # 3. CALCULATE POINTS, GD, GF, GA FROM SEASON 2025
    # ---------------------------------------------------------
    for _, row in season_df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = row['FTHG'], row['FTAG']
        
        # Track Matches Played
        stats[home]['Matches'] += 1
        stats[away]['Matches'] += 1
        
        # Track Goals
        stats[home]['GF_Total'] += hg
        stats[home]['GA_Total'] += ag
        stats[away]['GF_Total'] += ag
        stats[away]['GA_Total'] += hg
        
        # Track GD
        stats[home]['GD'] += (hg - ag)
        stats[away]['GD'] += (ag - hg)
        
        # Track Points
        if hg > ag: # Home Win
            stats[home]['Points'] += 3
        elif ag > hg: # Away Win
            stats[away]['Points'] += 3
        else: # Draw
            stats[home]['Points'] += 1
            stats[away]['Points'] += 1
            
        # Track Date
        stats[home]['Last_Match_Date'] = row['Date']
        stats[away]['Last_Match_Date'] = row['Date']

    # ---------------------------------------------------------
    # 4. FINALIZE AVERAGES & REST
    # ---------------------------------------------------------
    final_stats = {}
    
    for team, data in stats.items():
        # Calculate Averages
        matches = max(1, data['Matches']) # Avoid div by zero
        avg_gf = round(data['GF_Total'] / matches, 2)
        avg_ga = round(data['GA_Total'] / matches, 2)
        
        # Calculate Rest Days
        rest_days = (current_date - data['Last_Match_Date']).days
        rest_days = min(max(rest_days, 3), 14) # Clip between 3 and 14 to match training logic
        
        # Build clean dict
        final_stats[team] = {
            'Points': int(data['Points']),
            'GD': int(data['GD']),
            'Elo': int(data['Elo']),
            'Form': round(data['Form'], 2),
            'Rest': int(rest_days),
            'Avg_GF': avg_gf,
            'Avg_GA': avg_ga
        }
        
    return final_stats

# ---------------------------------------------------------
# EXECUTE
# ---------------------------------------------------------
# Usage: Pass your main 'df_poc' dataframe
initial_stats = build_current_standings(history_df)

# Check the result
print("\n--- CALCULATED STANDINGS ---")
df_check = pd.DataFrame.from_dict(initial_stats, orient='index').sort_values('Points', ascending=False)
print(df_check)

# --- STEP 4: FETCH FIXTURES & SIMULATE ---
print("4. Fetching upcoming fixtures...")
fixtures = fetch_api_data(['SCHEDULED', 'TIMED', 'POSTPONED'])

# If API returns empty (season over), handle it
if fixtures.empty:
    print("   No scheduled games found.")
else:
    print(f"   Simulating {len(fixtures)} remaining games...")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
N_SEASON_SIMS = 500  # Number of full seasons to simulate
current_standings = initial_stats  # Ensure this variable is defined from your previous block
#remaining_matches_base = pd.read_csv('epl_remain.csv')
remaining_matches_base = fixtures

print(f"Starting Monte Carlo Simulation: {N_SEASON_SIMS} Seasons...")

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS (Re-defined for clarity)
# ---------------------------------------------------------
def get_features_for_match(home, away, stats):
    h_stats = stats.get(home, {'Points': 20, 'GD': 0, 'Elo': 1500, 'Form': 1.3, 'Rest': 7})
    a_stats = stats.get(away, {'Points': 20, 'GD': 0, 'Elo': 1500, 'Form': 1.3, 'Rest': 7})
    
    features = pd.DataFrame([{
        'Home_Season_PPG': h_stats['Points'] / 15, 
        'Away_Season_PPG': a_stats['Points'] / 15,
        'Home_Elo': h_stats['Elo'],
        'Away_Elo': a_stats['Elo'],
        'Elo_Diff': h_stats['Elo'] - a_stats['Elo'],
        'Home_Form_EMA': h_stats['Form'],
        'Away_Form_EMA': a_stats['Form'],
        'Form_Diff': h_stats['Form'] - a_stats['Form'],
        'Rest_Diff': 0, 
        'Home_Avg_GF': 1.5 + (h_stats['Elo'] - 1500)/500,
        'Away_Avg_GF': 1.2 + (a_stats['Elo'] - 1500)/500,
        'Att_vs_Def': 0.2,
        'Poisson_H': 0.45, 
        'Poisson_D': 0.25,
        'Poisson_A': 0.30
    }])
    return features

def update_table(home, away, result_char, stats):
    if result_char == 'H':
        stats[home]['Points'] += 3
        stats[home]['GD'] += 1
        stats[home]['Elo'] += 5
        stats[away]['Elo'] -= 5
    elif result_char == 'A':
        stats[away]['Points'] += 3
        stats[away]['GD'] += 1
        stats[away]['Elo'] += 10
        stats[home]['Elo'] -= 10
    else: # 'D'
        stats[home]['Points'] += 1
        stats[away]['Points'] += 1
    return stats

# ---------------------------------------------------------
# 3. MONTE CARLO LOOP (The "Multiverse")
# ---------------------------------------------------------
# We will store the final points of every team in every simulation
all_sim_results = {team: [] for team in current_standings.keys()}
title_winners = []
relegated_teams = []

for sim_id in range(N_SEASON_SIMS):
    if sim_id % 50 == 0: print(f"  > Simulating Season {sim_id}/{N_SEASON_SIMS}...")
    
    # Create a fresh copy of the league table for this timeline
    sim_stats = copy.deepcopy(initial_stats) 
    
    # Loop through every remaining game
    for _, row in remaining_matches_base.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        # Skip unknown teams
        if home not in sim_stats or away not in sim_stats: continue
            
        # A. Get Features
        feats = get_features_for_match(home, away, sim_stats)
        
        # B. Get Ensemble Probabilities
        # XGBoost
        xgb_probs = best_model.predict_proba(feats)[0]
        # Logistic Regression
        feats_scaled = scaler.transform(feats)
        lr_probs = lr_model.predict_proba(feats_scaled)[0]
        
        # Blend (70% XGB / 30% LR)
        final_probs = (xgb_probs * 0.7) + (lr_probs * 0.3)
        final_probs /= final_probs.sum()
        
        # C. ROLL THE DICE (Probabilistic Outcome)
        # We pick one result based on the odds
        result_code = np.random.choice([0, 1, 2], p=final_probs)
        code_map = {0: 'A', 1: 'D', 2: 'H'}
        
        # D. Update the Table
        sim_stats = update_table(home, away, code_map[result_code], sim_stats)
        
    # --- SEASON FINISHED ---
    # 1. Convert stats to DataFrame to find ranks
    final_table = pd.DataFrame.from_dict(sim_stats, orient='index')
    final_table = final_table.sort_values(by=['Points', 'GD'], ascending=False)
    
    # 2. Record the stats
    ranked_teams = final_table.index.tolist()
    
    # Store Winner
    title_winners.append(ranked_teams[0])
    
    # Store Relegation (Bottom 3)
    relegated_teams.extend(ranked_teams[-3:])
    
    # Store Points for Averages
    for team, stats in sim_stats.items():
        all_sim_results[team].append(stats['Points'])

# ---------------------------------------------------------
# 4. ANALYZE RESULTS
# ---------------------------------------------------------
print("\n--- Predicted RESULTS ---")

# Calculate Average Points and Title Odds
summary_data = []
for team, points_list in all_sim_results.items():
    avg_pts = sum(points_list) / len(points_list)
    title_wins = title_winners.count(team)
    relegation_count = relegated_teams.count(team)
    
    summary_data.append({
        'Team': team,
        'Avg_Points': round(avg_pts, 1),
        'Title_Prob_%': round((title_wins / N_SEASON_SIMS) * 100, 1),
        'Relegation_Prob_%': round((relegation_count / N_SEASON_SIMS) * 100, 1)
    })

# Create Summary DataFrame
df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('Avg_Points', ascending=False).reset_index(drop=True)
df_summary.index += 1

print(df_summary)

# --- STEP 5: EXPORT ---
print("5. Exporting to CSV...")
df_summary.to_csv(OUTPUT_FILE, index=False)
print(f"Done! Data saved to {OUTPUT_FILE}")





