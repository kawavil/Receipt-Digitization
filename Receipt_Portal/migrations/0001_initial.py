# Generated by Django 3.0.5 on 2021-01-17 18:55

from django.db import migrations, models
import djongo.models.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Receipts',
            fields=[
                ('id', djongo.models.fields.ObjectIdField(auto_created=True, primary_key=True, serialize=False)),
                ('hotel_name', models.CharField(default='', max_length=100)),
                ('bill_amount', models.FloatField(default=0.0)),
            ],
        ),
    ]
