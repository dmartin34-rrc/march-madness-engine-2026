from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="March Madness Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing API and loading data...")

# Load model
try:
    model = joblib.load('march_madness_model.pkl')
except FileNotFoundError:
    raise Exception("Model file not found. Run model_builder.py first.")

# Load teams
teams_df = pd.read_csv('data/MTeams.csv')
team_dict = pd.Series(teams_df.TeamName.values, index=teams_df.TeamID).to_dict()

# Load current season data
reg_season = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
current_season = reg_season['Season'].max() 
recent_games = reg_season[reg_season['Season'] == current_season]

# Load KenPom rankings
ordinals = pd.read_csv('data/MMasseyOrdinals.csv')
recent_ords = ordinals[ordinals['Season'] == current_season]
pom_ranks = recent_ords[recent_ords['SystemName'] == 'POM']
final_ranks = pom_ranks.loc[pom_ranks.groupby(['TeamID'])['RankingDayNum'].idxmax()]
rank_dict = pd.Series(final_ranks.OrdinalRank.values, index=final_ranks.TeamID).to_dict()

# Load tournament seeds
seeds = pd.read_csv('data/MNCAATourneySeeds.csv')
recent_seeds = seeds[seeds['Season'] == current_season].copy()
recent_seeds['SeedNum'] = recent_seeds['Seed'].apply(lambda x: int(x[1:3]))
seed_dict = pd.Series(recent_seeds.SeedNum.values, index=recent_seeds.TeamID).to_dict()

# Calculate advanced stats
win_cols = ['WTeamID', 'WScore', 'LScore', 'WOR', 'WDR', 'LOR', 'LDR', 'WFGM', 'WFGA', 'WFGM3', 'WTO', 'LTO', 'WFTA', 'LFTA']
win_stats = recent_games[win_cols].copy()
win_stats.columns = ['TeamID', 'Pts', 'OppPts', 'ORB', 'DRB', 'OppORB', 'OppDRB', 'FGM', 'FGA', 'FGM3', 'TO', 'OppTO', 'FTA', 'OppFTA']
win_stats['Wins'] = 1
win_stats['Games'] = 1

loss_cols = ['LTeamID', 'LScore', 'WScore', 'LOR', 'LDR', 'WOR', 'WDR', 'LFGM', 'LFGA', 'LFGM3', 'LTO', 'WTO', 'LFTA', 'WFTA']
loss_stats = recent_games[loss_cols].copy()
loss_stats.columns = ['TeamID', 'Pts', 'OppPts', 'ORB', 'DRB', 'OppORB', 'OppDRB', 'FGM', 'FGA', 'FGM3', 'TO', 'OppTO', 'FTA', 'OppFTA']
loss_stats['Wins'] = 0
loss_stats['Games'] = 1

combined = pd.concat([win_stats, loss_stats])
season_stats = combined.groupby('TeamID').agg({
    'Pts': 'mean', 'OppPts': 'mean', 'ORB': 'mean', 'DRB': 'mean', 'OppORB': 'mean', 'OppDRB': 'mean',
    'FGM': 'mean', 'FGA': 'mean', 'FGM3': 'mean', 'TO': 'mean', 'OppTO': 'mean', 'FTA': 'mean', 'OppFTA': 'mean',
    'Wins': 'sum', 'Games': 'sum'
}).reset_index()

season_stats['WinPct'] = season_stats['Wins'] / season_stats['Games']
season_stats['Possessions'] = season_stats['FGA'] - season_stats['ORB'] + season_stats['TO'] + (0.475 * season_stats['FTA'])
season_stats['NetRtg'] = ((season_stats['Pts'] / season_stats['Possessions']) * 100) - ((season_stats['OppPts'] / season_stats['Possessions']) * 100)
season_stats['RebMargin'] = (season_stats['ORB'] + season_stats['DRB']) - (season_stats['OppORB'] + season_stats['OppDRB'])
season_stats['eFG'] = (season_stats['FGM'] + 0.5 * season_stats['FGM3']) / season_stats['FGA']
season_stats['TOMargin'] = season_stats['OppTO'] - season_stats['TO']

season_stats['Rank'] = season_stats['TeamID'].map(rank_dict).fillna(150)
season_stats['Seed'] = season_stats['TeamID'].map(seed_dict).fillna(10)

stats_lookup = season_stats.set_index('TeamID').to_dict('index')
print("API initialization complete.")

class MatchupRequest(BaseModel):
    team1_id: int
    team2_id: int

@app.get("/teams")
def get_teams():
    active_teams = [{'id': k, 'name': team_dict.get(k, "Unknown")} for k in stats_lookup.keys()]
    return sorted(active_teams, key=lambda x: x['name'])

@app.post("/predict")
def predict_matchup(req: MatchupRequest):
    if req.team1_id not in stats_lookup or req.team2_id not in stats_lookup:
        raise HTTPException(status_code=404, detail="Team stats not found.")

    t1 = stats_lookup[req.team1_id]
    t2 = stats_lookup[req.team2_id]

    # Feature set 1: Team 1 vs Team 2
    f1 = [[
        t1['WinPct'] - t2['WinPct'],
        t1['NetRtg'] - t2['NetRtg'],
        t1['RebMargin'] - t2['RebMargin'],
        t1['eFG'] - t2['eFG'],
        t1['TOMargin'] - t2['TOMargin'],
        t1['Rank'] - t2['Rank'],
        t1['Seed'] - t2['Seed']
    ]]

    # Feature set 2: Team 2 vs Team 1
    f2 = [[
        t2['WinPct'] - t1['WinPct'],
        t2['NetRtg'] - t1['NetRtg'],
        t2['RebMargin'] - t1['RebMargin'],
        t2['eFG'] - t1['eFG'],
        t2['TOMargin'] - t1['TOMargin'],
        t2['Rank'] - t1['Rank'],
        t2['Seed'] - t1['Seed']
    ]]

    # Get probabilities for both directions
    prob1 = model.predict_proba(f1)[0] # [Prob Team 2 wins, Prob Team 1 wins]
    prob2 = model.predict_proba(f2)[0] # [Prob Team 1 wins, Prob Team 2 wins]

    # Average the probability that Team 1 wins
    # (prob1[1] is T1 winning in f1, prob2[0] is T1 winning in f2)
    avg_prob_t1 = (prob1[1] + prob2[0]) / 2
    avg_prob_t2 = 1 - avg_prob_t1

    if avg_prob_t1 > avg_prob_t2:
        winner_name = team_dict.get(req.team1_id)
        confidence = avg_prob_t1
    else:
        winner_name = team_dict.get(req.team2_id)
        confidence = avg_prob_t2

    # Return the averaged results
    return {
        "team1_name": team_dict.get(req.team1_id),
        "team2_name": team_dict.get(req.team2_id),
        "predicted_winner_name": winner_name,
        "confidence": round(float(confidence) * 100, 2),
        "team1_stats": {
            "Seed": int(t1['Seed']), "Rank": int(t1['Rank']), "WinPct": round(t1['WinPct'] * 100, 1),
            "NetRtg": round(t1['NetRtg'], 1), "RebMargin": round(t1['RebMargin'], 1),
            "eFG_Pct": round(t1['eFG'] * 100, 1), "TO_Margin": round(t1['TOMargin'], 1)
        },
        "team2_stats": {
            "Seed": int(t2['Seed']), "Rank": int(t2['Rank']), "WinPct": round(t2['WinPct'] * 100, 1),
            "NetRtg": round(t2['NetRtg'], 1), "RebMargin": round(t2['RebMargin'], 1),
            "eFG_Pct": round(t2['eFG'] * 100, 1), "TO_Margin": round(t2['TOMargin'], 1)
        }
    }
