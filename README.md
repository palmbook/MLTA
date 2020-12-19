# MLTA
Machine Learning Driven Technical Analysis Library in Python.

## Motivation
Technical Analysis has long been a stable for generating insights for trading. While the predictive power of each individual indicator is up for a debate, it is no denying that technical indicators are useful tools to create summary statistics for time-series data. This usefulness is not limited to finance, but rather any domain with necessity to deal with time series.

However, most indicators are interpreted in a too simplistic way. For example, candlestick patterns classify each candle into one of a couple classes. However, we would get more useful information if we can also measure how close a candle matches each class.

MLTA enables this possibility. Instead of returning classes, MLTA generates the likelihood of a candle belonging to a given class. The result is more granula information which leads to an improvement in accuracy.

## Installation

MLTA is not yet on pypi. You can install the package through the following command:

pip install git+https://github.com/palmbook/MLTA.git

## Usage

#### Candlestick Patterns

	from MLTA import Candlestick

	candle_model = Candlestick()

	# df must have columns open, high, low, and close (in lower letters)
	# class_prob contains columns of probabilities for each possible class
	class_prob = candle_model.CDL3INSIDE(df)

Available methods are

* CDL3INSIDE
* CDL3LINESTRIKE
* CDL3OUTSIDE
* CDL3WHITESOLDIERS
* CDLCLOSINGMARUBOZU
* CDLCOUNTERATTACK
* CDLDOJI
* CDLDRAGONFLYDOJI
* CDLGAPSIDESIDEWHITE
* CDLGRAVESTONEDOJI
* CDLHAMMER
* CDLHARAMI
* CDLHOMINGPIGEON
* CDLINVERTEDHAMMER
* CDLLADDERBOTTOM
* CDLLONGLEGGEDDOJI
* CDLLONGLINE
* CDLMARUBOZU
* CDLMATCHINGLOW
* CDLMORNINGDOJISTAR
* CDLMORNINGSTAR
* CDLRICKSHAWMAN
* CDLRISEFALL3METHODS
* CDLSEPARATINGLINES
* CDLSHORTLINE
* CDLSTICKSANDWICH
* CDLTAKURI
* CDLTASUKIGAP
* CDLUNIQUE3RIVER
* bullishPin
* bearishPin
