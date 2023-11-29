name = 'MLTA'

import numpy as np
import pandas as pd

import joblib

import ta

from catboost import CatBoostClassifier

import os

class Candlestick:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(module_dir, 'candle_models.joblib')
    candle_models = joblib.load(file_path)
    
    def checkDF(self, df):
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        
        assert df.shape[0] > 0
        
    def patternProb(self, df, pattern):
        self.checkDF(df)
        
        df = df.copy()
        divisor = df[['open', 'high', 'low', 'close']].mean(axis=1)
        df = df[['open', 'high', 'low', 'close']].div(divisor, axis=0)
        
        y_pred = self.candle_models[pattern].predict_proba(df)
        
        return pd.DataFrame(y_pred, columns = [pattern + '_class_' + str(c) for c in self.candle_models[pattern].classes_], index=df.index)
        
    def CDL3INSIDE(self, df):
        return self.patternProb(df, 'CDL3INSIDE')
        
    def CDL3LINESTRIKE(self, df):
        return self.patternProb(df, 'CDL3LINESTRIKE')
        
    def CDL3OUTSIDE(self, df):
        return self.patternProb(df, 'CDL3OUTSIDE')
        
    def CDL3WHITESOLDIERS(self, df):
        return self.patternProb(df, 'CDL3WHITESOLDIERS')
        
    def CDLCLOSINGMARUBOZU(self, df):
        return self.patternProb(df, 'CDLCLOSINGMARUBOZU')
        
    def CDLCOUNTERATTACK(self, df):
        return self.patternProb(df, 'CDLCOUNTERATTACK')
        
    def CDLDOJI(self, df):
        return self.patternProb(df, 'CDLDOJI')
        
    def CDLDRAGONFLYDOJI(self, df):
        return self.patternProb(df, 'CDLDRAGONFLYDOJI')
        
    def CDLGAPSIDESIDEWHITE(self, df):
        return self.patternProb(df, 'CDLGAPSIDESIDEWHITE')
        
    def CDLGRAVESTONEDOJI(self, df):
        return self.patternProb(df, 'CDLGRAVESTONEDOJI')
        
    def CDLHAMMER(self, df):
        return self.patternProb(df, 'CDLHAMMER')
        
    def CDLHARAMI(self, df):
        return self.patternProb(df, 'CDLHARAMI')
        
    def CDLHOMINGPIGEON(self, df):
        return self.patternProb(df, 'CDLHOMINGPIGEON')
    
    def CDLINVERTEDHAMMER(self, df):
        return self.patternProb(df, 'CDLINVERTEDHAMMER')
    
    def CDLLADDERBOTTOM(self, df):
        return self.patternProb(df, 'CDLLADDERBOTTOM')
        
    def CDLLONGLEGGEDDOJI(self, df):
        return self.patternProb(df, 'CDLLONGLEGGEDDOJI')
        
    def CDLLONGLINE(self, df):
        return self.patternProb(df, 'CDLLONGLINE')
        
    def CDLMARUBOZU(self, df):
        return self.patternProb(df, 'CDLMARUBOZU')
        
    def CDLMATCHINGLOW(self, df):
        return self.patternProb(df, 'CDLMATCHINGLOW')
        
    def CDLMORNINGDOJISTAR(self, df):
        return self.patternProb(df, 'CDLMORNINGDOJISTAR')
        
    def CDLMORNINGSTAR(self, df):
        return self.patternProb(df, 'CDLMORNINGSTAR')
        
    def CDLRICKSHAWMAN(self, df):
        return self.patternProb(df, 'CDLRICKSHAWMAN')
        
    def CDLRISEFALL3METHODS(self, df):
        return self.patternProb(df, 'CDLRISEFALL3METHODS')
        
    def CDLSEPARATINGLINES(self, df):
        return self.patternProb(df, 'CDLSEPARATINGLINES')
        
    def CDLSHORTLINE(self, df):
        return self.patternProb(df, 'CDLSHORTLINE')
        
    def CDLSTICKSANDWICH(self, df):
        return self.patternProb(df, 'CDLSTICKSANDWICH')
        
    def CDLTAKURI(self, df):
        return self.patternProb(df, 'CDLTAKURI')
        
    def CDLTASUKIGAP(self, df):
        return self.patternProb(df, 'CDLTASUKIGAP')
        
    def CDLUNIQUE3RIVER(self, df):
        return self.patternProb(df, 'CDLUNIQUE3RIVER')
        
    def bullishPin(self, df):
        return self.patternProb(df, 'bullishPin')
        
    def bearishPin(self, df):
        return self.patternProb(df, 'bearishPin')

class AdvancedIndicators:

    @staticmethod
    def fibonacciMovingAverage(df, p=10):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        assert p <= 15, 'p can only be equal to or less than 15.'
        assert p > 1, 'p can only be more than 1.'
        
        df = df.copy()
        
        sequence = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
        
        total = df['close'].rolling(2).mean()
        
        for i in range(1, p):
            total = total + df['close'].rolling(sequence[i]).mean()
        
        return total / p
    
    @staticmethod
    def countdownIndicator(df, p=8):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        assert 'open' in df.columns, 'Column "open" must be in dataframe.'
        assert 'high' in df.columns, 'Column "high" must be in dataframe.'
        assert 'low' in df.columns, 'Column "low" must be in dataframe.'
        assert p > 1, 'p can only be more than 1.'
        
        df = df.copy()
        
        df['upside1'] = df['close'] > df['open']
        df['upside2'] = df['high'] > df['high'].shift(1)
        
        df['downside1'] = df['close'] < df['open']
        df['downside2'] = df['low'] < df['low'].shift(1)
        
        df['cup'] = df['upside1'].rolling(p).sum() + df['upside2'].rolling(p).sum()
        df['cdp'] = df['downside1'].rolling(p).sum() + df['downside2'].rolling(p).sum()
        
        return (df['cup'] - df['cdp']).rolling(3).mean()
       
    @staticmethod
    def supplyDemandIndicator(df):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        assert 'open' in df.columns, 'Column "open" must be in dataframe.'
        assert 'high' in df.columns, 'Column "high" must be in dataframe.'
        assert 'low' in df.columns, 'Column "low" must be in dataframe.'
        
        return ((df['high'] - df[['open', 'close']].max(axis=1)) - (df[['open', 'close']].min(axis=1) - df['low'])) / df['close']
        
    @staticmethod
    def superTrend(df, p=14, mult=2):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        assert 'high' in df.columns, 'Column "high" must be in dataframe.'
        assert 'low' in df.columns, 'Column "low" must be in dataframe.'
        
        df = df.copy()
        
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=p)
        
        df['basicub'] = ((df['high'] + df['low']) / 2) + (mult * df['atr'])
        df['basiclb'] = ((df['high'] + df['low']) / 2) - (mult * df['atr'])
        
        finalub = list()
        finallb = list()
        for i in range(df.shape[0]):
            if (i == 0):
                finalub.append(df['basicub'].iloc[i])
                finallb.append(df['basiclb'].iloc[i])
            else:
                if (df['basicub'].iloc[i] < finalub[i - 1]) | (df['close'].iloc[i - 1] > finalub[i - 1]):
                    finalub.append(df['basicub'].iloc[i])
                else:
                    finalub.append(finalub[i - 1])
                
                if (df['basiclb'].iloc[i] > finallb[i - 1]) | (df['close'].iloc[ i - 1] < finallb[i  -1]):
                    finallb.append(df['basiclb'].iloc[i])
                else:
                    finallb.append(finallb[i - 1])
        
        supertrend = list()
        for i in range(df.shape[0]):
            if (i == 0):
                supertrend.append(finalub[i])
            else:
                if (supertrend[i - 1] == finalub[i - 1]) & (df['close'].iloc[i] <= finalub[i]):
                    supertrend.append(finalub[i])
                elif (supertrend[i - 1] == finalub[i - 1]) & (df['close'].iloc[i] > finalub[i]):
                    supertrend.append(finallb[i])
                elif (supertrend[i - 1] == finallb[i - 1]) & (df['close'].iloc[i] >= finallb[i]):
                    supertrend.append(finallb[i])
                else:
                    supertrend.append(finalub[i])
        
        df['SuperTrend'] = supertrend
        
        return df['SuperTrend']
    @staticmethod
    def RainbowIndex(df):
        df = df.copy()
        
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        
        df = df[['close']].copy()
        
        for i in range(2, 201, 2):
            df['sma' + str(i)] = df['close'] > df['close'].rolling(i).mean()
        
        df['score'] = df.drop(['close'], axis=1).mean(axis=1)
        
        return df['score']
        
    @staticmethod
    def RainbowIndexLT(df):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        
        df = df[['close']].copy()
        
        df['RainbowIndexLT'] = (df['close'] > df['close'].rolling(10).mean()).astype('int')
        total = 1
        
        for i in range(20, 500, 10):
            df['RainbowIndexLT'] = df['RainbowIndexLT'] + (df['close'] > df['close'].rolling(i).mean()).astype('int')
            total += 1
        
        df['RainbowIndexLT'] = df['RainbowIndexLT'] / total
        
        return df['RainbowIndexLT']
        
    @staticmethod
    def KlingerOscillator(df, return_all=False):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        assert 'high' in df.columns, 'Column "high" must be in dataframe.'
        assert 'low' in df.columns, 'Column "low" must be in dataframe.'
        assert 'volume' in df.columns, 'Column "volume" must be in dataframe.'
        
        df = df.copy()
            
        df['KTrend'] = -1 + (2 * ((df['high'] + df['low'] + df['close']) >
            (df['high'] + df['low'] + df['close']).shift(1)).astype('int'))
        df['KDM'] = df['high'] - df['low']
        
        cm = list()
        for i in range(0, df.shape[0]):
            if i == 0:
                cm.append(df['KDM'].iloc[i])
            else:
                if df['KTrend'].iloc[i] == df['KTrend'].iloc[i-1]:
                    lastcm = cm[-1]
                    cm.append(lastcm + df['KDM'].iloc[i])
                else:
                    cm.append(df['KDM'].iloc[i-1] + df['KDM'].iloc[i])
        df['KCM'] = cm
        
        df['KVF'] = df['volume'] * (2 * ((df['KDM'] / df['KCM'].clip(lower=0.0001)) - 1)) * df['KTrend'] * 100
        
        df['KlingerOscillator'] = df['KVF'].ewm(span=34).mean() - df['KVF'].ewm(span=55).mean()
        
        if return_all:
            return df[['KTrend', 'KDM', 'KCM', 'KVF', 'KlingerOscillator']]
            
        return df['KlingerOscillator']
        
    @staticmethod
    def LogRegressionZScore(df, period=100):
        assert 'close' in df.columns, 'Column "close" must be in dataframe.'
        
        def LRZ(close):
            df = pd.DataFrame(close, columns=['close'])
            
            df['x'] = np.arange(df.shape[0])
            df['price_y']=np.log(df['close'])
            
            b,a =np.polyfit(df['x'],df['price_y'],1)
            
            df['priceTL']=b*df['x'] + a
            df['y-TL']=df['price_y']-df['priceTL']
            return df.iloc[-1]['y-TL'] / np.std(df['y-TL'])
        
        return df['close'].rolling(period).apply(LRZ)
        
class Transform:
    
    @staticmethod
    def fractionalDifferencing(series, d, window=20):
        def diffVector(d, window):
            w = 1
            vector = list()
            vector.append(w)

            for k in range(1, window):
                w = -w *( (d - k + 1) / k)
                vector.append(w)

            return np.array(vector)
        
        dvector = diffVector(d, window)
        
        def dot(a):
            return np.dot(a, dvector)
        
        result = np.log(series).rolling(window).apply(dot)
        
        return result
        
        