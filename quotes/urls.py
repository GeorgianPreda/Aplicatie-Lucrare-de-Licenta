from django.conf.urls.static import static
from django.urls import path
from . import views
from stocks import settings

urlpatterns = [
    path('', views.portfolio, name="portfolio"),
    path('home.html', views.home, name="home"),
    path('about.html', views.about, name="about"),
    path('portfolio.html', views.portfolio, name="portfolio"),
    path('printCAPM.html', views.printCAPM, name="printCAPM"),
    path('upload', views.upload, name="upload"),
    path('delete/<stock_id>', views.delete, name="delete"),
    path('printChart', views.printChart, name="printChart"),
    path('printRIDGE', views.printRIDGE, name="printRIDGE"),
    path('printLSTM', views.printLSTM, name="printLSTM"),
    # path('output_dr.html', views.output_dr, name="output_dr"),
]

if settings.DEBUG:
    urlpatterns == static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)