from django.http import HttpResponse
from django.shortcuts import render


class ReceiptInfo:

    def get(self, request):
        context = {}
        return render(request, 'demo.html', context)

    def post(self, request):
        context = {}
        return render(request, 'demo.html', context)
