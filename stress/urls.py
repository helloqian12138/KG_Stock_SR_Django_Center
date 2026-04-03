from django.urls import path
from . import views

urlpatterns = [
    path('', views.taxwar_page),
    path('taxwar', views.taxwar_page),
    path('taxwar/compare', views.taxwar_compare_page),
    path('macroshock', views.macroshock_page),
    path('ppi', views.ppi_page),
    path('api/fsi/history', views.api_fsi_history),
    path('api/predict', views.api_predict),
    path('api/macroshock/predict', views.api_macroshock_predict),
    path('api/ppi/predict', views.api_ppi_predict),
    path('api/taxwar/model-compare', views.api_taxwar_model_compare),
]
