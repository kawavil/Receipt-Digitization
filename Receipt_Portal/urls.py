from .views.index import Index
from django.urls import path


urlpatterns = [
    path('', Index.as_view(), name="home"),
    ]
