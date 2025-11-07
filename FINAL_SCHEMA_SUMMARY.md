# Final Database Schema - Complete Reference

## ✅ **COMPLETE SCHEMA (All Enhancements)**

---

## 📊 **option_chain_snapshots Table**

### **Complete Column List:**

| # | Column | Type | Example | Description |
|---|--------|------|---------|-------------|
| 1 | id | INTEGER | 12345 | Primary key (auto) |
| 2 | **timestamp** | TIMESTAMP | 2025-01-06 14:30:00 | When captured |
| 3 | **exchange** | TEXT | NSE, BSE | Exchange identifier |
| 4 | **strike** | REAL | 25600, 83500 | Strike price |
| 5 | **option_type** | TEXT | CE, PE | Call or Put |
| 6 | **symbol** | TEXT | NIFTY25N0425600CE | Trading symbol |
| 7 | **oi** | INTEGER | 1,215,600 | Open Interest |
| 8 | **ltp** | REAL | 223.6 | Last Traded Price |
| 9 | token | INTEGER | 12345678 | Instrument token |
| 10 | **underlying_price** | REAL | 25597, 83459 | ✅ NIFTY/SENSEX price |
| 11 | **moneyness** | TEXT | ITM, ATM, OTM | ✅ Strike classification |
| 12 | **pct_change_5m** | REAL | 4.5, -2.3 | ✅ 5-min OI % change |
| 13 | **pct_change_10m** | REAL | 8.2, -5.1 | ✅ 10-min OI % change |
| 14 | **pct_change_15m** | REAL | 12.5, -8.7 | ✅ 15-min OI % change |
| 15 | **pct_change_30m** | REAL | 18.9, -15.2 | ✅ 30-min OI % change |
| 16 | created_at | TIMESTAMP | Auto | Record creation |

**Total: 16 columns** (bold = key columns for analysis)

---

## 🎯 **Sample Record (Complete)**

### **NIFTY 25600 CE at 14:30:00 (NIFTY = 25,597)**

```json
{
  "id": 12345,
  "timestamp": "2025-01-06 14:30:00",
  "exchange": "NSE",
  "strike": 25600,
  "option_type": "CE",
  "symbol": "NIFTY25N0425600CE",
  "oi": 1215600,
  "ltp": 258.3,
  "token": 12345678,
  "underlying_price": 25597,    ← NIFTY price at this moment
  "moneyness": "ATM",           ← Classified as ATM
  "pct_change_5m": 0.0,
  "pct_change_10m": 0.0,
  "pct_change_15m": null,
  "pct_change_30m": null,
  "created_at": "2025-01-06 14:30:00"
}
```

---

## 📈 **Moneyness Classification Logic**

### **For CALL Options (CE):**

| Condition | Moneyness | Example (NIFTY = 25,597) |
|-----------|-----------|--------------------------|
| Strike < Underlying | **ITM** | 25,500 CE → ITM |
| Strike = ATM Strike | **ATM** | 25,600 CE → ATM |
| Strike > Underlying | **OTM** | 25,650 CE → OTM |

### **For PUT Options (PE):**

| Condition | Moneyness | Example (NIFTY = 25,597) |
|-----------|-----------|--------------------------|
| Strike > Underlying | **ITM** | 25,650 PE → ITM |
| Strike = ATM Strike | **ATM** | 25,600 PE → ATM |
| Strike < Underlying | **OTM** | 25,500 PE → OTM |

---

## 📊 **Complete CSV Export Example**

### **After Fresh Start, CSV Will Show:**

```csv
timestamp,exchange,strike,option_type,symbol,oi,ltp,underlying_price,moneyness,pct_change_5m,pct_change_10m,pct_change_15m,pct_change_30m
2025-01-06 14:30:00,NSE,25500,CE,NIFTY25N0425600CE,1215600,335.8,25597,ITM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25500,PE,NIFTY25N0425600PE,2992875,47.6,25597,OTM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25550,CE,NIFTY25N0425550CE,243150,294.8,25597,ITM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25550,PE,NIFTY25N0425550PE,2307975,58.5,25597,OTM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25600,CE,NIFTY25N0425600CE,93000,258.3,25597,ATM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25600,PE,NIFTY25N0425600PE,747600,71.3,25597,ATM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25650,CE,NIFTY25N0425650CE,58050,223.6,25597,OTM,0.0,0.0,,
2025-01-06 14:30:00,NSE,25650,PE,NIFTY25N0425650PE,784650,86.6,25597,ITM,0.0,0.0,,
2025-01-06 14:30:00,BSE,83500,CE,SENSEX...,154380,611.5,83459,ATM,0.0,,,
2025-01-06 14:30:00,BSE,83500,PE,SENSEX...,885720,79.2,83459,ATM,0.0,,,
```

**Notice:**
- ✅ underlying_price populated (25597 for NSE, 83459 for BSE)
- ✅ moneyness shows ITM/ATM/OTM
- ✅ Same underlying_price for all options at same timestamp
- ✅ Moneyness changes based on option_type (CE vs PE)

---

## 🎯 **Complete Save Process**

### **What Happens Every 30 Seconds:**

```python
# For NSE at 14:30:00

# Step 1: Get current data
underlying_price = 25597  # NIFTY LTP
atm_strike = 25600        # Calculated ATM

# Step 2: Prepare call options
call_options = [
    {
        'strike': 25500,
        'symbol': 'NIFTY25N0425500CE',
        'latest_oi': 1215600,
        'ltp': 335.8,
        'token': 12345678,
        'strike_type': 'itm',  # Already calculated
        'pct_changes': {'5m': 0.0, '10m': 0.0, ...}
    },
    {
        'strike': 25600,
        'strike_type': 'atm',  # ATM strike
        ...
    },
    {
        'strike': 25650,
        'strike_type': 'otm',  # Above underlying
        ...
    }
    # ... 11 total
]

# Step 3: Save to database
db.save_option_chain_snapshot(
    exchange='NSE',
    call_options=call_options,
    put_options=put_options,
    underlying_price=25597,  # Saved with EACH record
    atm_strike=25600,
    timestamp=datetime(2025, 1, 6, 14, 30, 0)
)

# Step 4: Database receives 22 records
# Each with: strike, oi, ltp, underlying_price, moneyness, pct_changes
```

---

## 🔍 **Query Examples with Moneyness**

### **1. Get Only ATM Options**

```sql
SELECT timestamp, exchange, strike, option_type, oi, ltp, underlying_price
FROM option_chain_snapshots
WHERE moneyness = 'ATM'
  AND DATE(timestamp) = DATE('now')
ORDER BY timestamp DESC;
```

### **2. Compare ITM vs OTM Behavior**

```sql
SELECT moneyness,
       AVG(oi) as avg_oi,
       AVG(ltp) as avg_ltp,
       AVG(pct_change_10m) as avg_10m_change
FROM option_chain_snapshots
WHERE DATE(timestamp) = DATE('now')
GROUP BY moneyness;
```

### **3. ATM Option Price Movements**

```sql
SELECT timestamp, strike, option_type, ltp, underlying_price, pct_change_10m
FROM option_chain_snapshots
WHERE moneyness = 'ATM'
  AND exchange = 'NSE'
  AND DATE(timestamp) = DATE('now')
ORDER BY timestamp;
```

### **4. Find High Activity by Moneyness**

```sql
SELECT moneyness, COUNT(*) as count
FROM option_chain_snapshots
WHERE ABS(pct_change_10m) > 10.0
  AND DATE(timestamp) = DATE('now')
GROUP BY moneyness;

Result:
moneyness | count
----------|------
ATM       | 45    ← ATM has most activity!
ITM       | 23
OTM       | 12
```

---

## 💡 **ML Training Benefits**

### **Better Feature Engineering:**

```python
# Moneyness as categorical feature
df = pd.get_dummies(df, columns=['moneyness'])

Features now include:
- moneyness_ITM   (0 or 1)
- moneyness_ATM   (0 or 1)
- moneyness_OTM   (0 or 1)
```

### **Stratified Analysis:**

```python
# Train separate models for each moneyness
for money in ['ITM', 'ATM', 'OTM']:
    df_subset = df[df['moneyness'] == money]
    model = train_model(df_subset)
    print(f"{money} model accuracy: {model.score()}")
```

**Expected:** ATM options likely have different patterns than ITM/OTM!

---

## 🎊 **Final Database Structure**

### **Every 30 Seconds, Each Option Record Contains:**

✅ **Basic Info:**
- Timestamp
- Exchange (NSE/BSE)
- Strike
- Option Type (CE/PE)
- Symbol
- Token

✅ **Market Data:**
- OI (Open Interest)
- LTP (Option Price)
- **Underlying Price** (NIFTY/SENSEX)

✅ **Classification:**
- **Moneyness** (ITM/ATM/OTM)

✅ **OI Changes:**
- 5-minute % change
- 10-minute % change
- 15-minute % change
- 30-minute % change

**Total: 16 fields per option record!**

---

## 🚀 **Action Steps**

### **1. Delete All Records:**
```bash
python delete_all_records.py
# Type: DELETE ALL
```

### **2. Restart Application:**
```bash
python oi_tracker_web.py
# Enter 2FA
```

### **3. Wait 10-15 Minutes**
Let data accumulate with new schema

### **4. Export & Verify:**
```bash
python export_to_csv.py
# Choose option 2
```

### **5. Open CSV in Excel:**
You'll see all columns including:
- ✅ underlying_price
- ✅ moneyness (ITM/ATM/OTM)

---

## 📋 **Complete Save Operation**

```
Every 30 seconds (per exchange):

NSE Thread:
├── Calculate: ATM = 25600, NIFTY = 25597
├── For each Call option (11 strikes):
│   ├── Strike 25500: moneyness = 'ITM' (< ATM)
│   ├── Strike 25550: moneyness = 'ITM' (< ATM)
│   ├── Strike 25600: moneyness = 'ATM' (= ATM) ← Highlighted!
│   ├── Strike 25650: moneyness = 'OTM' (> ATM)
│   └── Save: {strike, oi, ltp, underlying_price=25597, moneyness, pct_changes}
│
├── For each Put option (11 strikes):
│   ├── Strike 25500: moneyness = 'OTM' (< ATM for PE)
│   ├── Strike 25550: moneyness = 'OTM' (< ATM for PE)
│   ├── Strike 25600: moneyness = 'ATM' (= ATM) ← Highlighted!
│   ├── Strike 25650: moneyness = 'ITM' (> ATM for PE)
│   └── Save: {strike, oi, ltp, underlying_price=25597, moneyness, pct_changes}
│
└── Total: 22 records saved with complete data

BSE Thread:
└── (Same process for SENSEX with underlying_price = 83459)
```

---

## ✅ **Verification After Restart**

### **Check Schema:**
```bash
sqlite3 oi_tracker.db
.schema option_chain_snapshots
```

**Should show:**
```sql
CREATE TABLE option_chain_snapshots (
    ...
    underlying_price REAL,
    moneyness TEXT,
    pct_change_5m REAL,
    ...
);
```

### **Check Data (After 10 Minutes):**
```bash
python view_database.py
```

**Should show:**
```
strike | option_type | moneyness | underlying_price | ltp
-------|-------------|-----------|------------------|------
25500  | CE          | ITM       | 25597            | 335.8
25500  | PE          | OTM       | 25597            | 47.6
25600  | CE          | ATM       | 25597            | 258.3
25600  | PE          | ATM       | 25597            | 71.3
25650  | CE          | OTM       | 25597            | 223.6
25650  | PE          | ITM       | 25597            | 86.6
```

---

## 🎯 **What You Now Have**

**Complete Option Chain Data with:**

1. ✅ Timestamp (when captured)
2. ✅ Exchange (NSE/BSE)
3. ✅ Strike price
4. ✅ Option type (CE/PE)
5. ✅ Trading symbol
6. ✅ Open Interest (raw value)
7. ✅ Option price (LTP)
8. ✅ **Underlying price** (NIFTY/SENSEX at that moment)
9. ✅ **Moneyness** (ITM/ATM/OTM classification)
10. ✅ **5-minute OI % change**
11. ✅ **10-minute OI % change**
12. ✅ **15-minute OI % change**
13. ✅ **30-minute OI % change**

**Perfect for:**
- ✅ Excel pivot tables
- ✅ SQL analysis
- ✅ ML training
- ✅ Strategy backtesting
- ✅ Pattern recognition
- ✅ Statistical analysis

---

## 🔥 **Ready to Clean and Restart!**

Run these commands in order:

```bash
# 1. Delete all old records
python delete_all_records.py
# Type: DELETE ALL

# 2. Start fresh
python oi_tracker_web.py
# Enter 2FA when prompted

# 3. Wait 10-15 minutes for data accumulation

# 4. Export and verify
python export_to_csv.py
# Choose option 2

# 5. Open CSV in Excel
# Verify: underlying_price and moneyness columns populated!
```

**Your database is now PERFECT for comprehensive analysis and ML training!** 🎊

