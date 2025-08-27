#!/usr/bin/env python3
"""
Fight Rankings with Elimination Details
=======================================
Shows all players, their ranks, and who eliminated them.
"""

import pandas as pd
import os

def show_rankings_with_kills():
    """
    Show all players, their rankings, and who killed them.
    """
    # Find most recent detailed rankings file
    simulations_dir = "simulations"
    detailed_files = []
    
    for filename in os.listdir(simulations_dir):
        if 'detailed_rankings_' in filename and filename.endswith('.csv'):
            filepath = os.path.join(simulations_dir, filename)
            detailed_files.append((filepath, os.path.getmtime(filepath)))
    
    if not detailed_files:
        print("Error: No detailed rankings files found!")
        return
    
    # Use most recent detailed rankings file
    detailed_files.sort(key=lambda x: x[1], reverse=True)
    latest_file = detailed_files[0][0]
    
    # Read and display rankings
    df = pd.read_csv(latest_file)
    df = df.sort_values('Rank')
    
    print("⚔️ FIGHT RANKINGS WITH ELIMINATIONS")
    print("="*45)
    print(f"Total Players: {len(df)}")
    print("="*45)
    
    for _, row in df.iterrows():
        rank = row['Rank']
        player = row['Player']
        eliminated_by = row['Eliminated_By']
        
        # Format rank with proper suffix
        if rank == 1:
            rank_str = f"{rank}st"
            emoji = "🥇"
        elif rank == 2:
            rank_str = f"{rank}nd" 
            emoji = "🥈"
        elif rank == 3:
            rank_str = f"{rank}rd"
            emoji = "🥉"
        elif rank <= 10:
            rank_str = f"{rank}th"
            emoji = "🏅"
        elif rank <= 20:
            rank_str = f"{rank}th"
            emoji = "⭐"
        else:
            rank_str = f"{rank}th"
            emoji = "❌"
        
        # Format elimination info
        if eliminated_by == 'WINNER':
            elimination_info = "- WINNER"
        else:
            elimination_info = f"- killed by {eliminated_by}"
        
        print(f"{rank_str:<4} {player:<25} {emoji} {elimination_info}")
    
    print("="*45)
    winner = df.iloc[0]
    last_place = df.iloc[-1]
    print(f"🏆 Winner: {winner['Player']}")
    print(f"💀 First Eliminated: {last_place['Player']} (killed by {last_place['Eliminated_By']})")

if __name__ == "__main__":
    show_rankings_with_kills()
