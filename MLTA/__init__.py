name = 'MLTA'

import numpy as np
import pandas as pd

import joblib

from catboost import CatBoostClassifier

import os

class Candlestick:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(module_dir, 'MLTA/candle_models.joblib')
    candle_models = joblib.load(file_path)
    
    def checkDF(df):
        assert 'Open' in df.columns
        assert 'High' in df.columns
        assert 'Low' in df.columns
        assert 'Close' in df.columns
        
        assert df.shape[0] > 0
        
    def patternProb(df, pattern):
        checkDF(df)
        
        df = df.copy()
        divisor = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        df = df[['Open', 'High', 'Low', 'Close']].div(divisor, axis=0)
        
        y_pred = candle_models[pattern].predict_proba(df)
        
        return pd.DataFrame(y_pred, columns = [pattern + '_class_' + str(c) for c in candle_models[pattern].classes_], index=df.index)
        
    def CDL3INSIDE(df):
        return patternProb(df, 'CDL3INSIDE')
        
    def CDL3LINESTRIKE(df):
        return patternProb(df, 'CDL3LINESTRIKE')
        
    def CDL3OUTSIDE(df):
        return patternProb(df, 'CDL3OUTSIDE')
        
    def CDL3WHITESOLDIERS(df):
        return patternProb(df, 'CDL3WHITESOLDIERS')
        
    def CDLCLOSINGMARUBOZU(df):
        return patternProb(df, 'CDLCLOSINGMARUBOZU')
        
    def CDLCOUNTERATTACK(df):
        return patternProb(df, 'CDLCOUNTERATTACK')
        
    def CDLDOJI(df):
        return patternProb(df, 'CDLDOJI')
        
    def CDLDRAGONFLYDOJI(df):
        return patternProb(df, 'CDLDRAGONFLYDOJI')
        
    def CDLGAPSIDESIDEWHITE(df):
        return patternProb(df, 'CDLGAPSIDESIDEWHITE')
        
    def CDLGRAVESTONEDOJI(df):
        return patternProb(df, 'CDLGRAVESTONEDOJI')
        
    def CDLHAMMER(df):
        return patternProb(df, 'CDLHAMMER')
        
    def CDLHARAMI(df):
        return patternProb(df, 'CDLHARAMI')
        
    def CDLHOMINGPIGEON(df):
        return patternProb(df, 'CDLHOMINGPIGEON')
    
    def CDLINVERTEDHAMMER(df):
        return patternProb(df, 'CDLINVERTEDHAMMER')
    
    def CDLLADDERBOTTOM(df):
        return patternProb(df, 'CDLLADDERBOTTOM')
        
    def CDLLONGLEGGEDDOJI(df):
        return patternProb(df, 'CDLLONGLEGGEDDOJI')
        
    def CDLLONGLINE(df):
        return patternProb(df, 'CDLLONGLINE')
        
    def CDLMARUBOZU(df):
        return patternProb(df, 'CDLMARUBOZU')
        
    def CDLMATCHINGLOW(df):
        return patternProb(df, 'CDLMATCHINGLOW')
        
    def CDLMORNINGDOJISTAR(df):
        return patternProb(df, 'CDLMORNINGDOJISTAR')
        
    def CDLMORNINGSTAR(df):
        return patternProb(df, 'CDLMORNINGSTAR')
        
    def CDLRICKSHAWMAN(df):
        return patternProb(df, 'CDLRICKSHAWMAN')
        
    def CDLRISEFALL3METHODS(df):
        return patternProb(df, 'CDLRISEFALL3METHODS')
        
    def CDLSEPARATINGLINES(df):
        return patternProb(df, 'CDLSEPARATINGLINES')
        
    def CDLSHORTLINE(df):
        return patternProb(df, 'CDLSHORTLINE')
        
    def CDLSTICKSANDWICH(df):
        return patternProb(df, 'CDLSTICKSANDWICH')
        
    def CDLTAKURI(df):
        return patternProb(df, 'CDLTAKURI')
        
    def CDLTASUKIGAP(df):
        return patternProb(df, 'CDLTASUKIGAP')
        
    def CDLUNIQUE3RIVER(df):
        return patternProb(df, 'CDLUNIQUE3RIVER')
