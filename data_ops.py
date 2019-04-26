import fix_yahoo_finance as yf

def get_datasets(stock_name):
    if (stock_name == "Amazon"):
        ticker = yf.Ticker("AMZN")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Google"):
        ticker = yf.Ticker("GOOG")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Apple"):
        ticker = yf.Ticker("AAPL")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Microsoft"):
        ticker = yf.Ticker("MSFT")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "AAL"):
        ticker = yf.Ticker("AAL")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Tesla"):
        ticker = yf.Ticker("TSLA")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Bitcoin"):
        ticker = yf.Ticker("BTC-USD")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]

    elif (stock_name== "Aut"):
        symbols = ["VOE.VI"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "MixedTech"):
        symbols = ["AMZN","TSLA","AAPL","GOOG","MSFT"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "Mixed"):
        symbols = ["GE","VWAGY","BMWYY","F"]
        #symbols = ["TSLA","AMD"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "MixedCrypto"):
        symbols = ["BTC-USD","ETH-USD"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets
    else:
        ticker = yf.Ticker(stock_name)
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]