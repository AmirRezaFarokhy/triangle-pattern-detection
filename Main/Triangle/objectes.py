import pandas as pd
import numpy as np
import talib

class Indicators:
	def __init__(self, opens, lows, highs, closes):
		self.close = closes
		self.open = opens
		self.low = lows
		self.high = highs

	# Returns ATR values
	def AverageTrueRange(self, number_range=14, ema=True):
		tr = np.amax(np.vstack(((self.high-self.low).to_numpy(), (abs(self.high-self.close)).to_numpy(), (abs(self.low-self.close)).to_numpy())).T, axis=1)
		return pd.Series(tr).rolling(number_range).mean().to_numpy()

	def RSI(self, periods=14, ema=True):
		close_delta = self.close.diff()

		up = close_delta.clip(lower=0)
		down = -1 * close_delta.clip(upper=0)

		if ema==True:
			ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
			ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
		else:
			ma_up = up.rolling(window=periods, adjust=False).mean()
			ma_down = down.rolling(window=periods, adjust=False).mean()

		rsi = ma_up / ma_down
		rsi = 100 - (100/(1 + rsi))
		return rsi


	def Moving_Average(self, days=200):
		moving_averages = self.close.rolling(days).mean()
		return moving_averages


	def Bollian_Band(self, std=2, days=20):
		MA = self.close.rolling(window=days).mean()
		STD = self.close.rolling(window=days).std()
		Upper = MA + std*STD
		Lower = MA - std*STD
		data.dropna(inplace=True)
		Upper = data["Upper"]
		Lower = data["Lower"]
		data.drop("STD", 1, inplace=True)
		return Upper, Lower


	def stocastic(self, df ,k_period=14, d_period=3, col="BTC"):
		df['n_high'] = self.high.rolling(k_period).max()
		df['n_low'] = self.low.rolling(k_period).min()
		df['%K'] = (self.close-df['n_low'])*100/(df['n_high']-df['n_low'])
		Dstoc = df['%K'].rolling(d_period).mean()
		Kstoc = df['%K']
		return Kstoc, Dstoc


	def MACD(self, price, slow, fast, smooth):
		exp1 = price.ewm(span=fast, adjust=False).mean()
		exp2 = price.ewm(span=slow, adjust=False).mean()
		macd = pd.DataFrame(exp1 - exp2).rename(columns={'close':'macd'})
		signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(
															columns={'macd':'signal'})
		hist = pd.DataFrame(macd['macd']-signal['signal']).rename(columns={0:'hist'})
		#frames =  [macd, signal, hist]
		#df = pd.concat(frames, join='inner', axis=1)
		return macd, signal, hist


class SupportVSResistanced(Indicators):

	def __init__(self, dfs):
		self.main_df = dfs

	def support(self, low, n_before, n_after):
		for i in range(low-n_before+1, low+1):
			if self.main_df["low"].iloc[i]>self.main_df["low"].iloc[i-1]:
				return False

		for i in range(low+1, low+n_after+1):
			if self.main_df["low"].iloc[i]<self.main_df["low"].iloc[i-1]:
				return False

		return True

	def resistance(self, low, n_before, n_after):
		for i in range(low-n_before+1, low+1):
			if self.main_df["high"].iloc[i]<self.main_df["high"].iloc[i-1]:
				return False

		for i in range(low+1, low+n_after+1):
			if self.main_df["high"].iloc[i]>self.main_df["high"].iloc[i-1]:
				return False

		return True


	def isFarFromLevel(self, l, data):
		delta = np.mean(self.main_df["high"]-self.main_df["low"])
		if np.sum([abs(l-x)<delta for x in data])==0:
			return True
		else:
			return False
			


class PriceActionChandles:

	def __init__(self, dataframe):
		self.main_df = dataframe
		self.chanles_pattern_name = []
		self.count = 0

	def Pin_Bar(self):
		self.chanles_pattern_name.append("Pin_Bar")
		self.count += 1
		self.main_df["Pin_Bar"] = talib.CDLHAMMER(self.main_df["open"], self.main_df["high"], 
												self.main_df["low"], self.main_df["close"]) 

		return self.main_df, self.chanles_pattern_name

	def piercing_line(self):
		self.chanles_pattern_name.append("PiercingLine")
		self.count += 1
		self.main_df["PiercingLine"] = talib.CDLPIERCING(self.main_df["open"], self.main_df["high"], 
														self.main_df["low"], self.main_df["close"])
		return self.main_df, self.chanles_pattern_name

	def engolfing(self):
		self.chanles_pattern_name.append("Engolfing")
		self.count += 1
		self.main_df["Engolfing"] = talib.CDLENGULFING(self.main_df["open"], self.main_df["high"], 
													self.main_df["low"], self.main_df["close"])

		return self.main_df, self.chanles_pattern_name

	def evening_star(self):
		self.chanles_pattern_name.append("EveningStar")
		self.count += 1
		self.main_df["EveningStar"] = talib.CDLEVENINGSTAR(self.main_df["open"], self.main_df["high"], 
									self.main_df["low"], self.main_df["close"])

		return self.main_df, self.chanles_pattern_name

	def dragonfly_doji(self):
		self.chanles_pattern_name.append("DragonflyDoji")
		self.count += 1
		self.main_df["DragonflyDoji"] = talib.CDLDRAGONFLYDOJI(self.main_df["open"], 
															   self.main_df["high"], 
															   self.main_df["low"], 
															   self.main_df["close"])
		return  self.main_df, self.chanles_pattern_name

	def grave_stone_doji(self):
		self.chanles_pattern_name.append("GraveStoneDoji")
		self.count += 1
		self.main_df["GraveStoneDoji"] = talib.CDLGRAVESTONEDOJI(self.main_df["open"], 
															   self.main_df["high"], 
															   self.main_df["low"], 
															   self.main_df["close"])
		return  self.main_df, self.chanles_pattern_name

	def three_line_strike(self):
		self.chanles_pattern_name.append("three_line_strike")
		self.count += 1
		self.main_df["three_line_strike"] = talib.CDL3LINESTRIKE(self.main_df["open"], 
																 self.main_df["high"], 
																 self.main_df["low"], 
																 self.main_df["close"])	 
		return self.main_df, self.chanles_pattern_name

	def dark_cloud_cover(self):
		self.chanles_pattern_name.append("dark_cloud_cover")
		self.count += 1
		self.main_df["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(self.main_df["open"], 
																	self.main_df["high"], 
																	self.main_df["low"], 
																	self.main_df["close"])	 
		return self.main_df, self.chanles_pattern_name

	def tasukigap(self):
		self.chanles_pattern_name.append("tasukigap")
		self.count += 1
		self.main_df["tasukigap"] = talib.CDLTASUKIGAP(self.main_df["open"], 
																 self.main_df["high"], 
																 self.main_df["low"], 
																 self.main_df["close"])	 
		return self.main_df, self.chanles_pattern_name


	def three_black_crow(self):
		self.chanles_pattern_name.append("three_black_crow")
		self.count += 1
		self.main_df["three_black_crow"] = talib.CDL3BLACKCROWS(self.main_df["open"], 
																 self.main_df["high"], 
																 self.main_df["low"], 
																 self.main_df["close"])	 
		return self.main_df, self.chanles_pattern_name


