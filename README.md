# OI Tracker Web Application

A beautiful, real-time web-based Open Interest tracker for NIFTY/BANKNIFTY/SENSEX options with lightning-fast updates via WebSocket.

## Features

✨ **Real-time Updates**: WebSocket-powered live data streaming  
🎨 **Modern UI**: Beautiful gradient design with glassmorphism effects  
📊 **Dual View**: Side-by-side CALL and PUT options tables  
🚨 **Smart Alerts**: Automatic highlighting of significant OI changes  
⚡ **Fast Refresh**: 30-second update intervals (configurable)  
📱 **Responsive**: Works on desktop, tablet, and mobile  

## Screenshots

- **Modern Dark Theme** with gradient backgrounds
- **Color-coded Strikes**: ITM (Green), ATM (Cyan), OTM (Red)
- **Highlighted Alerts**: Automatically highlights cells exceeding thresholds
- **Live Status Badge**: Shows connection status with pulsing animation

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Configure Credentials

Edit `oi_tracker_web.py` and update:

```python
USER_ID = "YOUR_ZERODHA_ID"    # Your Zerodha Login ID
PASSWORD = "YOUR_PASSWORD"      # Your Zerodha Password
```

**Note**: 2FA code will be requested when you start the application.

### 3. Configure Trading Parameters

Adjust these settings in `oi_tracker_web.py`:

```python
# For NIFTY 50
UNDERLYING_SYMBOL = "NIFTY 50"
STRIKE_DIFFERENCE = 50
OPTIONS_COUNT = 5

# For BANK NIFTY (uncomment and comment NIFTY settings)
# UNDERLYING_SYMBOL = "NIFTY BANK"
# STRIKE_DIFFERENCE = 100
# OPTIONS_COUNT = 5

# For SENSEX (uncomment and adjust exchanges)
# UNDERLYING_SYMBOL = "SENSEX"
# STRIKE_DIFFERENCE = 100
# EXCHANGE_NFO_OPTIONS = "BFO"
# EXCHANGE_LTP = "BSE"
```

## Running the Application

### 1. Start the Server

```bash
python oi_tracker_web.py
```

You will be prompted to enter your 2FA code:

```
OI TRACKER WEB APPLICATION
============================================================

Login ID: YOUR_ID
Enter your 2FA PIN or TOTP code: ******
```

### 2. Access the Web Interface

Once initialized, open your web browser and navigate to:

```
http://localhost:5000
```

### 3. View Live Data

The interface will automatically:
- Connect via WebSocket
- Subscribe to underlying and option tokens
- Display real-time OI changes
- Highlight significant changes based on thresholds

## Configuration Options

### Update Frequency

```python
REFRESH_INTERVAL_SECONDS = 30  # Update every 30 seconds
```

### Alert Thresholds

```python
PCT_CHANGE_THRESHOLDS = {
    5: 8.0,   # Highlight if 5-min % change > 8%
    10: 10.0, # Highlight if 10-min % change > 10%
    15: 15.0, # Highlight if 15-min % change > 15%
    30: 25.0  # Highlight if 30-min % change > 25%
}
```

### Historical Data Window

```python
HISTORICAL_DATA_MINUTES = 40  # Fetch 40 minutes of historical data
```

### Options Coverage

```python
OPTIONS_COUNT = 5  # Shows 5 ITM + ATM + 5 OTM strikes
```

## Architecture

### Backend (Flask + SocketIO)
- `Flask`: Web framework
- `Flask-SocketIO`: WebSocket support for real-time updates
- `kite_trade`: Alternative Zerodha connection (no API needed)
- `KiteTicker`: WebSocket for real-time price data

### Frontend
- **Bootstrap 5**: Responsive layout
- **Socket.IO Client**: WebSocket connection
- **Custom CSS**: Modern glassmorphism design
- **Vanilla JavaScript**: No framework dependencies

### Data Flow
1. **Initialization**: Authenticate → Get enctoken → Connect WebSocket
2. **Background Thread**: Continuously fetches OI data every 30 seconds
3. **WebSocket Broadcast**: Pushes updates to all connected clients
4. **Frontend Update**: JavaScript receives data and updates tables in real-time

## API Endpoints

### REST API
- `GET /` - Main web interface
- `GET /api/data` - Get current OI data (JSON)

### WebSocket Events
- `connect` - Client connected
- `disconnect` - Client disconnected
- `data_update` - Server pushes new OI data

## Table Columns

### CALL Options Table
| Column | Description |
|--------|-------------|
| Strike | Option strike price (color-coded) |
| Symbol | Trading symbol |
| Latest OI | Current Open Interest |
| OI Time | Timestamp of latest OI |
| OI %Chg (5m) | 5-minute percentage change |
| OI %Chg (10m) | 10-minute percentage change |
| OI %Chg (15m) | 15-minute percentage change |
| OI %Chg (30m) | 30-minute percentage change |

### PUT Options Table
Same columns as CALL options, but for PUT contracts.

## Color Coding

### Strike Prices
- 🟢 **Green (ITM)**: In-The-Money
  - For CALLS: Strikes below ATM
  - For PUTS: Strikes above ATM
- 🔵 **Cyan (ATM)**: At-The-Money (current ATM strike)
- 🔴 **Red (OTM)**: Out-The-Money
  - For CALLS: Strikes above ATM
  - For PUTS: Strikes below ATM

### Percentage Changes
- 🟢 **Green**: Positive change (OI increased)
- 🔴 **Red**: Negative change (OI decreased)
- 🚨 **Highlighted**: Exceeds configured threshold (with blinking animation)

## Troubleshooting

### Issue: WebSocket not connecting
**Solution**: Check firewall settings, ensure port 5000 is available

### Issue: No data showing (all N/A)
**Causes**:
1. Market is closed (no live data available)
2. No historical data for the time window
3. WebSocket not receiving ticks

**Solution**: 
- Check if market is open
- Verify WebSocket connection status
- Check browser console for errors

### Issue: Enctoken expired
**Solution**: Restart the application and re-enter 2FA code

### Issue: Tables not updating
**Solution**:
1. Check browser console for WebSocket errors
2. Verify background thread is running (check terminal logs)
3. Refresh the page

## Performance Tips

1. **Reduce Refresh Interval**: Lower `REFRESH_INTERVAL_SECONDS` for faster updates (min 15s recommended)
2. **Reduce Options Count**: Lower `OPTIONS_COUNT` to track fewer strikes
3. **Reduce History Window**: Lower `HISTORICAL_DATA_MINUTES` for faster processing

## Logs

Application logs are saved to: `oi_tracker_web.log`

View logs in real-time:
```bash
# PowerShell
Get-Content oi_tracker_web.log -Wait -Tail 50
```

## Stopping the Application

Press `Ctrl+C` in the terminal to stop the server gracefully.

## Advanced: Running on Network

To access from other devices on your network:

1. Find your computer's IP address
2. The application is already configured to listen on `0.0.0.0:5000`
3. Access from other devices using: `http://YOUR_IP:5000`

## Security Notes

⚠️ **Important**:
- This application runs locally on your machine
- Your credentials are stored in the Python file (keep it secure)
- 2FA is requested at runtime (more secure)
- WebSocket connections are NOT encrypted by default
- Don't expose this to the public internet without proper security

## Support

For issues or questions:
1. Check the `oi_tracker_web.log` file
2. Enable DEBUG logging by changing `FILE_LOG_LEVEL = "DEBUG"`
3. Check browser console for frontend errors

## License

This application is for personal use only. Ensure compliance with Zerodha's terms of service.

---

**Happy Trading! 📈**

