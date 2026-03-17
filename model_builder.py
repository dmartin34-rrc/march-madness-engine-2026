import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

print("Loading historical NCAA data...")
reg_season = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
tourney = pd.read_csv('data/MNCAATourneyDetailedResults.csv')
ordinals = pd.read_csv('data/MMasseyOrdinals.csv')
seeds = pd.read_csv('data/MNCAATourneySeeds.csv')

# Seeds
print("Extracting Selection Committee Seeds...")
seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds = seeds[['Season', 'TeamID', 'SeedNum']]

# Rankings
print("Extracting final regular-season KenPom rankings...")
pom_ranks = ordinals[ordinals['SystemName'] == 'POM']
final_ranks = pom_ranks.loc[pom_ranks.groupby(['Season', 'TeamID'])['RankingDayNum'].idxmax()]
final_ranks = final_ranks[['Season', 'TeamID', 'OrdinalRank']]

# Advanced Efficiency Metrics
def get_season_averages(df):
    win_cols = ['Season', 'WTeamID', 'WScore', 'LScore', 'WOR', 'WDR', 'LOR', 'LDR', 'WFGM', 'WFGA', 'WFGM3', 'WTO', 'LTO', 'WFTA', 'LFTA']
    win_stats = df[win_cols].copy()
    win_stats.columns = ['Season', 'TeamID', 'Pts', 'OppPts', 'ORB', 'DRB', 'OppORB', 'OppDRB', 'FGM', 'FGA', 'FGM3', 'TO', 'OppTO', 'FTA', 'OppFTA']
    win_stats['Wins'] = 1
    win_stats['Games'] = 1
    
    loss_cols = ['Season', 'LTeamID', 'LScore', 'WScore', 'LOR', 'LDR', 'WOR', 'WDR', 'LFGM', 'LFGA', 'LFGM3', 'LTO', 'WTO', 'LFTA', 'WFTA']
    loss_stats = df[loss_cols].copy()
    loss_stats.columns = ['Season', 'TeamID', 'Pts', 'OppPts', 'ORB', 'DRB', 'OppORB', 'OppDRB', 'FGM', 'FGA', 'FGM3', 'TO', 'OppTO', 'FTA', 'OppFTA']
    loss_stats['Wins'] = 0
    loss_stats['Games'] = 1
    
    combined = pd.concat([win_stats, loss_stats])
    
    agg_funcs = {
        'Pts': 'mean', 'OppPts': 'mean', 'ORB': 'mean', 'DRB': 'mean', 'OppORB': 'mean', 'OppDRB': 'mean',
        'FGM': 'mean', 'FGA': 'mean', 'FGM3': 'mean', 'TO': 'mean', 'OppTO': 'mean', 'FTA': 'mean', 'OppFTA': 'mean',
        'Wins': 'sum', 'Games': 'sum'
    }
    season_stats = combined.groupby(['Season', 'TeamID']).agg(agg_funcs).reset_index()
    
    # 1. Win Percentage
    season_stats['WinPct'] = season_stats['Wins'] / season_stats['Games']
    
    # 2. Possessions & Net Rating
    season_stats['Possessions'] = season_stats['FGA'] - season_stats['ORB'] + season_stats['TO'] + (0.475 * season_stats['FTA'])
    season_stats['OffRtg'] = (season_stats['Pts'] / season_stats['Possessions']) * 100
    season_stats['DefRtg'] = (season_stats['OppPts'] / season_stats['Possessions']) * 100
    season_stats['NetRtg'] = season_stats['OffRtg'] - season_stats['DefRtg']
    
    # 3. Standard Advanced Metrics
    season_stats['RebMargin'] = (season_stats['ORB'] + season_stats['DRB']) - (season_stats['OppORB'] + season_stats['OppDRB'])
    season_stats['eFG'] = (season_stats['FGM'] + 0.5 * season_stats['FGM3']) / season_stats['FGA']
    season_stats['TOMargin'] = season_stats['OppTO'] - season_stats['TO'] 
    
    return season_stats[['Season', 'TeamID', 'WinPct', 'NetRtg', 'RebMargin', 'eFG', 'TOMargin']]

print("Crunching true possession-based efficiency stats...")
season_stats = get_season_averages(reg_season)

# Merge Rankings & Seeds
season_stats = pd.merge(season_stats, final_ranks, on=['Season', 'TeamID'], how='left')
season_stats['OrdinalRank'] = season_stats['OrdinalRank'].fillna(150)

season_stats = pd.merge(season_stats, seeds, on=['Season', 'TeamID'], how='left')
# If a team missed the tournament, give them a dummy seed of 10 for regular season diffs
season_stats['SeedNum'] = season_stats['SeedNum'].fillna(10)

# Build training data
print("Building historical tournament matchups...")
tourney = pd.merge(tourney, season_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney = tourney.rename(columns={
    'WinPct': 'W_WinPct', 'NetRtg': 'W_NetRtg', 'RebMargin': 'W_RebMargin', 'eFG': 'W_eFG', 'TOMargin': 'W_TOMargin', 'OrdinalRank': 'W_Rank', 'SeedNum': 'W_Seed'
})

tourney = pd.merge(tourney, season_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop'))
tourney = tourney.rename(columns={
    'WinPct': 'L_WinPct', 'NetRtg': 'L_NetRtg', 'RebMargin': 'L_RebMargin', 'eFG': 'L_eFG', 'TOMargin': 'L_TOMargin', 'OrdinalRank': 'L_Rank', 'SeedNum': 'L_Seed'
})

# Calculate Differentials
tourney['Diff_WinPct'] = tourney['W_WinPct'] - tourney['L_WinPct']
tourney['Diff_NetRtg'] = tourney['W_NetRtg'] - tourney['L_NetRtg']
tourney['Diff_RebMargin'] = tourney['W_RebMargin'] - tourney['L_RebMargin']
tourney['Diff_eFG'] = tourney['W_eFG'] - tourney['L_eFG']
tourney['Diff_TOMargin'] = tourney['W_TOMargin'] - tourney['L_TOMargin']
tourney['Diff_Rank'] = tourney['W_Rank'] - tourney['L_Rank']
tourney['Diff_Seed'] = tourney['W_Seed'] - tourney['L_Seed']

# Train XGBoost
print("Training the XGBoost Model...")

features = ['Diff_WinPct', 'Diff_NetRtg', 'Diff_RebMargin', 'Diff_eFG', 'Diff_TOMargin', 'Diff_Rank', 'Diff_Seed']
X = tourney[features].dropna()

X_flipped = -X.copy()
y = np.ones(len(X))
y_flipped = np.zeros(len(X_flipped))

X_final = pd.concat([X, X_flipped])
y_final = np.concatenate([y, y_flipped])

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Training Complete")
print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'march_madness_model.pkl')
