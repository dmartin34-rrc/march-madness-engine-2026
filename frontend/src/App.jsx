import { useState, useEffect } from 'react';

function StatRow({ label, val1, val2, unit = '', reverse = false }) {
  const t1Wins = reverse ? val1 < val2 : val1 > val2;
  const t2Wins = reverse ? val2 < val1 : val2 > val1;

  // Helper to format the display
  const format = (v) => {
    if (label === 'Tournament Seed' && v === 10) return 'N/A';
    if (label === 'National Rank') return `#${v}`;
    if (unit === '+') return v > 0 ? `+${v}` : v;
    if (unit === '%') return `${v}%`;
    return v;
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid #bfdbfe' }}>
      <div style={{ width: '30%', fontWeight: t1Wins ? 'bold' : 'normal', color: t1Wins ? '#16a34a' : '#1e293b' }}>
        {format(val1)}
      </div>
      <div style={{ width: '40%', fontSize: '14px', color: '#475569', fontWeight: '500' }}>{label}</div>
      <div style={{ width: '30%', fontWeight: t2Wins ? 'bold' : 'normal', color: t2Wins ? '#16a34a' : '#1e293b' }}>
        {format(val2)}
      </div>
    </div>
  );
}

function App() {
  const [teams, setTeams] = useState([]);
  const [team1, setTeam1] = useState('');
  const [team2, setTeam2] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('https://march-madness-engine-2026.onrender.com/teams')
      .then((res) => res.json())
      .then((data) => setTeams(data))
      .catch((err) => console.error("Error fetching teams:", err));
  }, []);

  const handlePredict = async () => {
    if (!team1 || !team2) {
      setError("Please select both teams.");
      return;
    }
    if (team1 === team2) {
      setError("Teams cannot be the same.");
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('https://march-madness-engine-2026.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ team1_id: parseInt(team1), team2_id: parseInt(team2) })
      });

      if (!response.ok) throw new Error("Matchup calculation failed.");
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getLogo = (name) => `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=2563eb&color=fff&size=100&bold=true`;

  return (
    <div style={{ maxWidth: '600px', margin: '50px auto', fontFamily: 'system-ui', textAlign: 'center' }}>
      <h1 style={{ lineHeight: '1.2', paddingBottom: '20px' }}>
        March Madness<br />Engine 2026
      </h1>

      <div style={{ display: 'flex', justifyContent: 'space-between', margin: '30px 0' }}>
        <select value={team1} onChange={(e) => setTeam1(e.target.value)} style={{ padding: '10px', fontSize: '16px', width: '45%', borderRadius: '5px' }}>
          <option value="">Select Team 1</option>
          {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>

        <span style={{ fontSize: '24px', fontWeight: 'bold', alignSelf: 'center', color: '#94a3b8' }}>VS</span>

        <select value={team2} onChange={(e) => setTeam2(e.target.value)} style={{ padding: '10px', fontSize: '16px', width: '45%', borderRadius: '5px' }}>
          <option value="">Select Team 2</option>
          {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      <button 
        onClick={handlePredict} disabled={loading}
        style={{ padding: '15px 30px', fontSize: '18px', fontWeight: 'bold', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', width: '100%', transition: '0.2s' }}
      >
        {loading ? "Calculating..." : "Simulate Matchup"}
      </button>

      {error && <p style={{ color: 'red', marginTop: '20px' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: '30px', padding: '25px', border: '2px solid #2563eb', borderRadius: '8px', backgroundColor: '#f8fafc', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}>
          <h2 style={{ color: '#1e293b', margin: '0 0 10px 0' }}>Projected Winner: <br/><span style={{ color: '#2563eb' }}>{result.predicted_winner_name}</span></h2>
          <p style={{ fontSize: '18px', margin: '0 0 20px 0', color: '#475569' }}>
            <strong>Model Confidence:</strong> {result.confidence}%
          </p>

          <hr style={{ margin: '20px 0', borderColor: '#e2e8f0', borderTop: 'none' }} />
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ width: '30%' }}>
                <img src={getLogo(result.team1_name)} alt={result.team1_name} style={{ borderRadius: '50%', width: '50px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }} />
                <div style={{ fontWeight: 'bold', marginTop: '5px', fontSize: '14px' }}>{result.team1_name}</div>
            </div>
            <div style={{ width: '40%', fontSize: '14px', color: '#94a3b8', fontWeight: 'bold' }}>ADVANCED METRICS</div>
            <div style={{ width: '30%' }}>
                <img src={getLogo(result.team2_name)} alt={result.team2_name} style={{ borderRadius: '50%', width: '50px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }} />
                <div style={{ fontWeight: 'bold', marginTop: '5px', fontSize: '14px' }}>{result.team2_name}</div>
            </div>
          </div>

          {/* Stat Comparison Section */}
          <StatRow label="Tournament Seed" val1={result.team1_stats.Seed} val2={result.team2_stats.Seed} reverse={true} />
          <StatRow label="National Rank" val1={result.team1_stats.Rank} val2={result.team2_stats.Rank} reverse={true} />
          <StatRow label="Win Percentage" val1={result.team1_stats.WinPct} val2={result.team2_stats.WinPct} unit="%" />
          <StatRow label="Net Rating" val1={result.team1_stats.NetRtg} val2={result.team2_stats.NetRtg} unit="+" />
          <StatRow label="Rebound Margin" val1={result.team1_stats.RebMargin} val2={result.team2_stats.RebMargin} unit="+" />
          <StatRow label="Effective FG%" val1={result.team1_stats.eFG_Pct} val2={result.team2_stats.eFG_Pct} unit="%" />
          <StatRow label="Turnover Margin" val1={result.team1_stats.TO_Margin} val2={result.team2_stats.TO_Margin} unit="+" />
        </div>
      )}
    </div>
  );
}

export default App;