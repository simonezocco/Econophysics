import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

Stock_Ticker = 'AAPL' # Ticker symbol for Apple Inc. = unique series of characters assigned to a company's stock
Days_Back = 252 # Number of days to look back for historical data
Years_to_Predict = 1
simulations = 50

# #yf.download returns a DataFrame with historical stock data.
prices = yf.download(Stock_Ticker, period=f"{Days_Back}d", interval= '1d')['Close'] #Download only Close Prices

# A DataFrame is a 2d labeled data structure with columns of potentially different types.
#Price	        Close	High	Low	Open	Volume
#Ticker	        AAPL	AAPL	AAPL	AAPL	AAPL
#Date					
#2024-06-06	193.167999	195.174376	192.860093	194.369843	41181800
#2024-06-07	195.561737	195.611403	192.830289	193.336843	53103900
#2024-06-10	191.817184	195.968994	190.853727	195.571683	97010200
#2024-06-11	205.752548	205.762490	192.323766	192.343620	172373300
#2024-06-12	211.632629	218.714520	205.504239	205.971070	198134300
#...	        ...	        ...	        ...	        ...	        ...

log_returns = np.log(prices / prices.shift(1)).dropna()
#Calculate log returns and dropna() function of pandas that removes rows or columns with NaN values (non numeric values)

dt = 1/252  # Time increment (1 trading day out of 252 trading days in a year)

mu = (log_returns.mean() / dt).item()
# Mean of log returns. .item() converts a single-value DataFrame to a classical python scalar
sigma = (log_returns.std() / np.sqrt(dt)).item()
S0 = prices.iloc[-1].item()  # Last closing price will be our first price in the simulation

steps = int(Years_to_Predict * 252)  # Number of trading days to simulate

W = np.random.standard_normal(size=(steps, simulations))
# Generate random numbers from a standard normal distribution for each step and simulation
exp = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W)

price_matrix = np.zeros((steps + 1, simulations)) # Initialize price matrix. +1 to include S0

price_matrix[0, :] = S0  # Set the first row to S0

for t in range(1, steps + 1):
    price_matrix[t, :] = price_matrix[t - 1, :] * exp[t - 1, :] # Simulate prices from S0

plt.figure(figsize=(10, 6)) #create a graphical figure with a specified size
historical_days = prices.index # Get the dates for historical prices
plt.plot(historical_days, prices, label='Historical Prices') # Plot historical prices. history_days is x-axis, prices is y-axis

last_historical_date = historical_days[-1]  # Last date in historical data
future_days = pd.date_range(start=last_historical_date, periods=steps + 1)[1:] # Ignore the first date which has price S0
plt.plot(future_days, price_matrix[1:,:], lw=1, alpha = 0.75) # Plot simulated prices, lw is line width. alpha is transparency
# price_matrix[1:,:] to ignore the first row which is S0
plt.title(f'Stochastic Simulations for {Stock_Ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=last_historical_date, color='red', linestyle='--', label='Start of Simulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()