from django.db import models
from djongo import models
# Create your models here.


class Receipts(models.Model):
    id = models.ObjectIdField(primary_key=True)
    emp_name = models.CharField(max_length=100, default="")
    service_name = models.CharField(max_length=100, default="")
    bill_amount = models.FloatField(default=0.0)

