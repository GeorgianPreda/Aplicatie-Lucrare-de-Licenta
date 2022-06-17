from multiprocessing import context

from django.shortcuts import render, redirect

from .capm import stocks_daily_return, beta, alpha, printcapm
from .dailyreturns import daily_return, stocks_df
from .lstm import df_predicted_lstm
from .ridge import df_predicted_ridge
from .models import Stock
from .forms import StockForm
from django.contrib import messages
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage

# Create your views here.
# .preprocessing import interactive_plot, stock_price_df
from .MLModels import *


def home(request):
    return render(request, 'home.html', {})


def about(request):
    import requests
    import json

    if request.method == 'POST':
        ticker = request.POST['ticker']
        # pk_987a02686f5440b79a62b4c032a315a5
        api_request = requests.get(
            "https://cloud.iexapis.com/stable/stock/" + ticker + "/quote?token=pk_987a02686f5440b79a62b4c032a315a5")
        try:
            api = json.loads(api_request.content)
        except Exception as e:
            api = "Error..."
        return render(request, 'about.html', {'api': api})
    else:
        api_request = requests.get(
            "https://cloud.iexapis.com/stable/stock/AMZN/quote?token=pk_987a02686f5440b79a62b4c032a315a5")
        api = json.loads(api_request.content)
        return render(request, 'about.html', {"api": api})


def portfolio(request):
    import requests
    import json

    if request.method == 'POST':
        form = StockForm(request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request, "Stock added successfully to the DB")
            return redirect('portfolio')
    else:
        ticker = Stock.objects.all()
        output = []
        for ticker_item in ticker:
            api_request = requests.get("https://cloud.iexapis.com/stable/stock/" + str(
                ticker_item) + "/quote?token=pk_987a02686f5440b79a62b4c032a315a5")
            try:
                api = json.loads(api_request.content)
                output.append(api)
            except Exception as e:
                api = "Error..."
        return render(request, 'portfolio.html', {'ticker': ticker, 'output': output})


def delete(request, stock_id):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request, "Ticker deleted successfully")
    return redirect(portfolio)


def printChart(request):
    plot = interactive_plot(stock_price_df, 'Stock Prices')
    return render(request, 'home.html', {"plot": plot})


def printRIDGE(request):
    # Plot the results
    # interactive_plot(df_predicted, "Original Vs. Prediction - Ridge Regression Model")
    plot = interactive_plot(df_predicted_ridge, "AAPL - Original Vs. Prediction - Ridge Regression Model")
    return render(request, 'home.html', {"plot": plot})


def printLSTM(request):
    # Plot the data
    # interactive_plot(df_predicted, "Original Vs Prediction - LSTM Model")
    plot = interactive_plot(df_predicted_lstm, " AAPL - Original Vs Prediction - LSTM Model")
    return render(request, 'home.html', {"plot": plot})


def printCAPM(request):
    import requests
    import json
    output_capm = printcapm()
    return render(request, 'printCAPM.html', {"output_capm": output_capm})


# def output_dr(request):
#     stocks_daily_return = print_dr()
#     return render(request, 'output_dr.html', {"stocks_daily_return": stocks_daily_return})

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        print(uploaded_file.name)
        name = uploaded_file.name
        print(uploaded_file.size)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
    return render(request, 'home.html', {"name": name})
