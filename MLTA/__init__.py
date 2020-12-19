name = 'MLTA'

import numpy as np
import pandas as pd

import joblib

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
