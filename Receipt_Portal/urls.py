from .views.index import Index
from .views.demo import ReceiptInfo
from django.urls import path


urlpatterns = [
    path('', Index.as_view(), name="home"),
    path('', ReceiptInfo.as_view(), name='receiptinfo'),
    ]
