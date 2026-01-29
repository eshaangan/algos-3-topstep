import subprocess
import time
import re
import sys
from datetime import datetime
import pandas as pd

# Configuration
VM_NAME = "topstep-trader-vm"
ZONE = "us-central1-a"
REFRESH_SECONDS = 10

def fetch_logs():
    """Fetch serial port output from GCP."""
    try:
        cmd = [
            "gcloud", "compute", "instances", "get-serial-port-output", 
            VM_NAME, "--zone", ZONE
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return ""

def parse_logs(log_text):
    """Parse raw logs into structured data."""
    lines = log_text.split('\n')
    
    signals = []
    trades = []
    status_updates = []
    app_started = False
    
    for line in lines:
        if "LIVE TRADING RUNNER - ML INTRADAY V3" in line:
            app_started = True
            
        if "Signal generated:" in line:
            try:
                parts = line.split('Signal generated:')
                meta = parts[0]
                content = parts[1]
                
                score_match = re.search(r'score=([\d\.]+)', content)
                score = float(score_match.group(1)) if score_match else 0.0
                
                signals.append({
                    'raw': line,
                    'score': score,
                    'action': 'BUY' if score > 0 else 'SELL',
                    'type': 'SIGNAL'
                })
            except:
                pass
                
        elif "Trade executed:" in line:
            trades.append(line)
            
        # Capture both DEBUG status and any other equity logs
        elif "Broker status:" in line or "Equity:" in line or "equity=" in line:
            try:
                # Regex for "Broker status: equity=123.45" OR "Equity: $ 123.45"
                equity_match = re.search(r'(?:equity=|Equity:\s*\$)\s*([\d\.,]+)', line, re.IGNORECASE)
                
                # Regex for "daily_pnl=123.45" OR "Daily P&L: $ 123.45"
                pnl_match = re.search(r'(?:daily_pnl=|Daily P&L:\s*(?:\\x1b\[\d+m)?\$)\s*([-\d\.,]+)', line, re.IGNORECASE)
                
                if equity_match:
                    eq_val = float(equity_match.group(1).replace(',', ''))
                    # If pnl matches, parse it; otherwise default to 0
                    if pnl_match:
                        pnl_str = pnl_match.group(1).replace(',', '')
                        # Handle potential ANSI color codes in value if any slipped through
                        pnl_val = float(re.sub(r'[^\d\.-]', '', pnl_str))
                    else:
                        pnl_val = 0.0
                    
                    status_updates.append({
                        'equity': eq_val,
                        'daily_pnl': pnl_val,
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                # Debug print only if needed
                # print(f"Parse error on line: {line} -> {e}")
                pass

    return signals, trades, status_updates, app_started

def clear_screen():
    print("\033[H\033[J", end="")

def main():
    print(f"Starting monitor for {VM_NAME}...")
    print("Press Ctrl+C to stop.")
    
    while True:
        logs = fetch_logs()
        signals, trades, statuses, app_started = parse_logs(logs)
        
        clear_screen()
        print("="*60)
        print(f"🤖 TOPSTEP TRADER MONITOR  |  {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        if not app_started:
            print("\n⏳ STATUS: VM is booting / Pulling Docker Image...")
            print("   (This takes ~2-3 minutes on fresh deploy)")
        elif statuses:
            latest = statuses[-1]
            print(f"\n💰 ACCOUNT STATUS")
            print(f"   Equity:    ${latest['equity']:,.2f}")
            print(f"   Daily P&L: ${latest['daily_pnl']:,.2f}")
        else:
            print("\n💰 STATUS: App running, waiting for first status update...")

        # Signals Section
        print(f"\n📡 LATEST SIGNALS (Last 5)")
        if signals:
            for s in signals[-5:]:
                # Try to extract time from raw line
                time_str = s['raw'][:19] if len(s['raw']) > 19 else ""
                print(f"   {time_str} | Score: {s['score']:.3f}")
        else:
            print("   No signals detected yet.")

        # Trades Section
        print(f"\n⚡ TRADES (Last 5)")
        if trades:
            for t in trades[-5:]:
                print(f"   {t.strip()}")
        else:
            print("   No trades executed yet.")
            
        print("\n" + "-"*60)
        print(f"Refreshing in {REFRESH_SECONDS} seconds...")
        time.sleep(REFRESH_SECONDS)

if __name__ == "__main__":
    main()
